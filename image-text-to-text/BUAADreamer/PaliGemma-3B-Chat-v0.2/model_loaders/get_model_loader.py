from environment import USE_TASK_MODEL_REGISTRY

print("Loading model...")

# default is to use the default model_loader
if not USE_TASK_MODEL_REGISTRY:
    # this is the default model loader that will run
    # NOTE a refactor of the existing github repo will be required in order to do this cleanly
    # we will simply rewrite server.py to import this file, and then move the original model_loader into this directory
    from model_loaders.model_loader import pipe

# if the model in question has a special architecture, we load custom code for that via an architecture based registry
# NOTE if we need to extend this further, we can introduce another env variable that acts as a special override for the registry
else:
    from model_loaders.architecture_registry.registry import Registry

    model_entity = Registry.get_model_entity()

    pipe = model_entity
