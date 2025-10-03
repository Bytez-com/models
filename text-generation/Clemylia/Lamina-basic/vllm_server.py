# NOTE important, makes vLLM think it has EiB's worth of GPU memory
import vllm_mocks
from vllm.entrypoints.openai.api_server import (
    cli_env_setup,
    run_server,
    uvloop,
    FlexibleArgumentParser,
    make_arg_parser,
    validate_parsed_serve_args,
)

if __name__ == "__main__":
    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)

    args = parser.parse_args()

    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
