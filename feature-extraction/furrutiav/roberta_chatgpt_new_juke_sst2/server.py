# NOTE this structuring is required because parallel model loading requires the start method to be set to spawn
if __name__ == "__main__" or __name__ == "server":
    from time import sleep
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
        FILES_SIZE_GB,
        MODEL_SIZE_GB,
        LOG_LOADING,
        DEVICE,
        SYSTEM_LOGS_PATH,
    )
    from loading_tracker import LoadingTracker
    import multiprocessing
    import threading
    from stats import SYSTEM_RAM_TRACKER

    inference_lock = threading.Lock()

    multiprocessing.set_start_method("spawn", force=True)  # Set start method to 'spawn'

    # Construct class for tracking downloading and loading models
    LOADING_TRACKER = LoadingTracker(
        task=TASK,
        model_id=MODEL_ID,
        device=DEVICE,
        files_size_in_GB=FILES_SIZE_GB,
        model_size_in_GB=MODEL_SIZE_GB,
        logging_enabled=LOG_LOADING,
    )

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
                app.logger.info(
                    f"Request to {request.path} with method {request.method}"
                )
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

            request_data = {}

            if request.method == "POST":
                try:
                    request_data = request.get_json()
                except Exception as exception:
                    app.logger.error(exception)

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

        @app.route("/", defaults={"path": ""})
        @app.route("/health", methods=["GET"])
        async def health_check():
            model_is_loaded = bool(LOADING_TRACKER.loading_is_done.value)

            model_failed_to_load = bool(LOADING_TRACKER.loading_failed.value)
            model_failed_to_load_exception = (
                LOADING_TRACKER.loading_failed_exception.value.decode("utf-8")
            )

            # NOTE if the model fails to load, the server crashes

            if model_failed_to_load:
                return (
                    f"Fatal, model failed to load with an exception:\n\n{model_failed_to_load_exception}",
                    503,
                )

            if model_is_loaded:
                return "ready", 200

            else:
                # NOTE this is the max timeout for our load balancer's healthcheck, which gives us more leeway with determining model readiness
                sleep(2)
                return "Model is not ready yet", 503

        @app.route("/load_model", methods=["GET"])
        def load_model():
            result = {"success": True, "exception": ""}

            def run_import():
                try:
                    from run_endpoint_handler import run_endpoint_handler  # noqa: F401
                except Exception:
                    exception = format_exc()
                    print("Model failed to load with an exception:\n\n", exception)

                    result["success"] = False
                    result["exception"] = exception

            thread = threading.Thread(target=run_import)

            thread.start()
            thread.join()

            status_code = 200 if result["success"] else 503

            return jsonify(result), status_code

        @app.route("/run", methods=["POST"])
        @track_analytics(event_name="Model Inference")
        def run():
            from run_endpoint_handler import run_endpoint_handler

            # NOTE important, pipeline is not thread safe, you do not want concurrent requests.
            # futhermore, oftentimes even though it will perform inference, it severely degrades performance. Throughput is higher in serial.
            with inference_lock:
                return run_endpoint_handler(request)

        @app.route("/status", methods=["GET"])
        async def load_status():
            progress_download = LOADING_TRACKER.percent_progress_download.value
            progress_load = LOADING_TRACKER.percent_progress_load.value

            # return response
            return jsonify(
                {
                    "progress_percent_download": progress_download,
                    "progress_percent_load": progress_load,
                    "download_done": bool(LOADING_TRACKER.downloading_is_done),
                    "done": bool(LOADING_TRACKER.loading_is_done.value),
                    "model_failed_to_load": bool(LOADING_TRACKER.loading_failed.value),
                    "model_failed_to_load_exception": (
                        LOADING_TRACKER.loading_failed_exception.value.decode("utf-8")
                    ),
                    "memory_stats": {
                        "device": DEVICE,
                        "MODEL_SIZE_GB": round(MODEL_SIZE_GB, 2),
                        "available_GB": round(LOADING_TRACKER.available_GB.value, 2),
                        "peak_GB": round(LOADING_TRACKER.peak_GB.value, 2),
                        "current_GB": round(LOADING_TRACKER.current_GB.value, 2),
                    },
                    "debug": {
                        "elapsed_time_s": LOADING_TRACKER.elapsed_time_in_seconds,
                    },
                },
            )

        @app.route("/logs", methods=["GET"])
        async def logs():
            try:
                with open(SYSTEM_LOGS_PATH, encoding="utf-8", errors="replace") as file:
                    logs = file.read()
                    return {"logs": logs, "logs_path": SYSTEM_LOGS_PATH}
            except Exception:
                exception = format_exc()
                return {"exception": exception}, 500

        @app.route("/stats/cpu/memory", methods=["GET"])
        def job_runner_cpu_memory():
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

        # Start tracking loading progress
        loading_tracker_process = multiprocessing.Process(
            target=LOADING_TRACKER.load_model_with_tracking,
            args=[f"http://localhost:{PORT}/load_model"],
        )

        loading_tracker_process.start()

        # NOTE this is job runner specific, find out how much memory is being used to run the program before loading the model
        # NOTE this is only accurate because the flask app is run from gunicorn, if we wanted debug to be more accurate
        # we'd want to do this + then loading the model AFTER the flask server starts in a thread
        SYSTEM_RAM_TRACKER.set_baseline_utilization_GB()

        # start the flask server
        if START_FLASK_DEBUG_SERVER:
            app.run(port=PORT, debug=False)

    except Exception as exception:
        stack_trace = format_exc()
        analytics(
            event_name="Model Error",
            request_props={"stackTrace": stack_trace},
        )

        raise exception
