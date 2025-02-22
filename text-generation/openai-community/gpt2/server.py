from datetime import datetime
from copy import deepcopy
from requests import request as httpRequest
import json
from traceback import format_exc
from flask import Flask, request, jsonify
import functools
from environment import (
    TASK,
    MODEL_ID,
    PORT,
    DISABLE_ANALYTICS,
    START_FLASK_DEBUG_SERVER,
    USE_PRODUCTION_ANALYTICS_ENDPOINT,
    API_KEY,
)
from stats import SYSTEM_RAM_TRACKER


######################### ANALYTICS LOGIC HERE #########################
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
        "modelName": MODEL_ID.split("/")[1],
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
            "modelName": MODEL_ID.split("/")[1],
            "task": TASK,
            "source": "container",
        },
    )

    app = Flask(__name__)

    # We want to keep our ordering the way it is
    app.json.sort_keys = False

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
        if request.method != "POST":
            app.logger.info(f"Request to {request.path} with method {request.method}")
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
                stack_trace=stack_trace,
            ),
            422,
        )

    @app.route("/health", methods=["GET"])
    async def health_check():
        return "", 200

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all():
        return "", 204

    @app.route("/run", methods=["POST"])
    @track_analytics(event_name="Model Inference")
    def run():
        return run_endpoint_handler(request)

    @app.route("/logs", methods=["GET"])
    def logs():
        log_file_path = "/var/log/cloud-init-output.log"

        with open(log_file_path, "r") as file:
            log_content = file.read()
        return jsonify({"log": log_content}), 200

    @app.route("/stats/cpu/memory", methods=["GET"])
    def load_status():
        stats = SYSTEM_RAM_TRACKER.get_ram_stats()

        peak_system_ram_usage_GB = stats["peak_system_ram_usage_GB"]
        peak_model_ram_usage_GB = stats["peak_model_ram_usage_GB"]

        # return response
        return jsonify(
            {
                "peak_system_ram_usage_GB": peak_system_ram_usage_GB,
                "peak_model_ram_usage_GB": peak_model_ram_usage_GB,
            },
        )

    # NOTE find out how much memory is being used to run the program before loading the model
    # NOTE this is only accurate because the flask app is run from gunicorn, if we wanted debug to be more accurate
    # we'd want to do this + then loading the model AFTER the flask server starts in a thread
    SYSTEM_RAM_TRACKER.set_baseline_utilization_GB()

    # NOTE model is loaded as a side effect of this import, this has to happen after all other setup logic
    from run_endpoint_handler import run_endpoint_handler

    if START_FLASK_DEBUG_SERVER:
        app.run(port=PORT, debug=False)

except Exception as exception:
    stack_trace = format_exc()
    analytics(
        event_name="Model Error",
        request_props={"stackTrace": stack_trace},
    )

    raise exception
