from datetime import datetime
from copy import deepcopy
from requests import request as httpRequest
import json
from os import environ
from traceback import format_exc
from flask import Flask, request, jsonify, send_file, Response
from model import model_run, model_eject
import functools
from utils import model_run_generator

#
MODEL_ID = environ.get("MODEL_ID", "")
TASK = environ.get("TASK", "")

######################### ANALYTICS LOGIC HERE #########################

# prevents calls to analytics when in the testing pipeline, this is set not in the Dockerfile, but as a docker run -e arg
DISABLE_ANALYTICS = json.loads(
    environ.get("DISABLE_ANALYTICS", 'false')
)

# Code is being run from gunicorn and not the debugger (flask will not start up)
START_FLASK_DEBUG_SERVER = json.loads(
    environ.get("START_FLASK_DEBUG_SERVER", "false")
)

# this should be set by the Dockerfile, this is to prevent the debugger
USE_PRODUCTION_ANALYTICS_ENDPOINT = json.loads(
    environ.get("USE_PRODUCTION_ANALYTICS_ENDPOINT", "false")
)

API_KEY = environ.get("KEY")
# 8002 to avoid conflicts with testing locally against local api server
PORT = environ.get("PORT", 8002)


VALIDATION_URL = (
    "https://api.bytez.com/containers/validation"
    if USE_PRODUCTION_ANALYTICS_ENDPOINT
    else "http://localhost:8000/containers/validation"
)

ANALYTICS_URL = (
    "https://api.bytez.com/containers/analytics"
    if USE_PRODUCTION_ANALYTICS_ENDPOINT
    else "http://localhost:8000/containers/analytics"
)


def raise_api_key_exception():
    raise Exception(
        """You must specify a Bytez api key for the environment variable KEY to use this container. 
                    Please visit: https://bytez.com to generate a key.
                    """
    )


def make_http_request(url, method="POST", data={}):
    response = httpRequest(
        url=url,
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "BytezModelRuntime/0.0.1",
        },
        data=json.dumps(data),
    )

    return response


def authorize(event_name, props):
    if DISABLE_ANALYTICS:
        return

    response = make_http_request(
        url=VALIDATION_URL,
        data={
            "eventName": event_name,
            "props": props,
            "apiKey": API_KEY,
            "created": datetime.now().isoformat(),
        },
    )

    if response.status_code != 204:
        raise_api_key_exception()

    pass


def analytics(event_name, request_props):
    if DISABLE_ANALYTICS:
        return
    # remember, props is a pointer
    request_props = deepcopy(request_props)

    props = {
        "modelId": MODEL_ID,
        "modelName": MODEL_ID.split('/')[1],
        "task": TASK,
        "source": "container",
        "requestProps": request_props,
    }

    try:
        response = make_http_request(
            url=ANALYTICS_URL,
            data={
                "apiKey": API_KEY,
                "eventName": event_name,
                "props": props,
                "created": datetime.now().isoformat(),
            },
        )

        # TODO in prod you should not see this
        if DISABLE_ANALYTICS:
            print(f"Analytics response: {response}")

    # do nothing on analytics failure
    # TODO in prod you should not see this
    except Exception as exception:
        if DISABLE_ANALYTICS:
            print("Analytics call failed: ", exception)


######################### CONTAINER BUSINESS LOGIC BEGINS HERE #########################
try:
    if not DISABLE_ANALYTICS and not API_KEY:
        raise_api_key_exception()

    # always attempt authorization on container startup
    authorize(
        event_name="Model Deploy",
        props={
            "modelId": MODEL_ID,
            "modelName": MODEL_ID.split('/')[1],
            "task": TASK,
            "source": "container",
        },
    )

    app = Flask(__name__)

    def track_analytics(event_name):
        def decorator(f):
            @functools.wraps(f)
            def decorated_function(*args, **kwargs):
                try:
                    request_data = request.get_json()
                    # API_KEY is a global const
                    analytics(event_name=event_name, request_props=request_data)

                    return f(
                        *args, **kwargs
                    )  # Proceed with the original function if auth succeeds
                except Exception as exception:
                    app.log_exception(exception)
                    raise exception

            return decorated_function

        return decorator

    @app.before_request
    def log_request():
        # skip healthcheck pings
        if request.path == "/health":
            return

        try:
            request_data = request.get_json()
            request_data_as_str = json.dumps(request_data, indent=4)
            app.logger.info(
                f"Request to {request.path} with method {request.method} and data:\n{request_data_as_str}"
            )
        except Exception as exception:
            app.logger.error(f"Error logging request: {exception}")

    @app.errorhandler(Exception)
    def handle_error(error):
        # Log error, or send it to a specific endpoint
        stack_trace = format_exc()
        app.logger.error(stack_trace)

        try:
            request_data = request.get_json()
        except Exception as exception:
            app.logger.error(exception)
            request_data = {}

        analytics(
            event_name="Model Error",
            request_props={**request_data, "stackTrace": stack_trace},
        )

        return (
            jsonify(
                error=str(error),
                # leaving stack trace out for now, we should have an argument validator function that
                # provides more insightful data to the user
                #    stack_trace=stack_trace
            ),
            422,
        )

    @app.route("/health", methods=["GET"])
    def health_check():
        return "", 200
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all():
        return "", 204

    @app.route("/eject", methods=["GET"])
    def eject():
        model_path = model_eject()

        return send_file(model_path)

    @app.route("/run", methods=["POST"])
    @track_analytics(event_name="Model Inference")
    def run():
        params = request.json.get("params", {})
        user_input = request.json["text"]
        stream = request.json.get("stream", False)

        if stream:
            output_generator = model_run_generator(user_input=user_input, params=params)

            return Response(
                output_generator(),
                content_type="text/event-stream; charset=utf-8",
            )

        # model inference
        model_output = model_run(user_input, params)

        # return response
        return jsonify({"output": model_output})
    
    if START_FLASK_DEBUG_SERVER:
        app.run(port=PORT, debug=False)

except Exception as exception:
    stack_trace = format_exc()
    analytics(
        event_name="Model Error",
        request_props={"stackTrace": stack_trace},
    )

    raise exception
