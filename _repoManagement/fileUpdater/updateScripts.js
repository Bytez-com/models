const { higherOrderIterator } = require("../higherOrderIterator");

const fs = require("fs").promises;
const pathModule = require("path");

const tasksToUpdate = [
  "text-generation",
  "audio-text-to-text",
  "image-text-to-text"
  // "video-text-to-text"
  //
  // "summarization",
  // "translation",
  // "text2text-generation",
  // "visual-question-answering",
  // "document-question-answering",
  // "depth-estimation",
  // "image-classification",
  // "object-detection",
  // "image-segmentation",
  // "text-to-image",
  // "image-to-text",
  // "unconditional-image-generation",
  // "video-classification",
  // "text-to-video",
  // "zero-shot-image-classification",
  // "mask-generation",
  // "zero-shot-object-detection",
  // "image-feature-extraction",
  // "text-classification",
  // "token-classification",
  // "question-answering",
  // "zero-shot-classification",
  // "translation",
  // "summarization",
  // "feature-extraction",
  // "text2text-generation",
  // "fill-mask",
  // "sentence-similarity",
  // "text-to-speech",
  // "text-to-audio",
  // "automatic-speech-recognition",
  // "audio-classification"
];

const filesToUpdate = [
  "adaptation.py",
  "download_bytez_repo.py",
  "environment.py",
  "model_loader.py",
  "run_endpoint_handler.py",
  "serve.sh",
  "streamer.py",
  "utils.py",
  "vllm_loader.py"
  // "model.py"

  // "environment.py"
  // "model_loader.py",
  // // "streamer.py"
  // // "utils.py",
  // "vllm_loader.py"
  // "vllm_server.py"
  // "vllm_mocks.py"
];

const BYTEZ_API_UTILITIES_PATH = pathModule.resolve(
  `${__dirname}/../../../../bytez-api-utilities`
);

// the "to" prop is relatively pathed
const filesToUpdateForSpecificTasks = {
  "audio-text-to-text": {
    filePaths: [
      // default model loader
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/model_loader.py`,
        to: "model_loader.py"
      },
      // endpoint handler
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/templates/multimodal/audio-text-to-text/containerFiles/run_endpoint_handler.py`,
        to: "run_endpoint_handler.py"
      },
      // task specific model_entity.py
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/audio_text_to_text/model_entity.py`,
        to: "architecture_registry_module/tasks/audio_text_to_text/model_entity.py"
      },
      // specific architecture updates
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/audio_text_to_text/architectures/Qwen2AudioForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/audio_text_to_text/architectures/Qwen2AudioForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/audio_text_to_text/architectures/Qwen2AudioForConditionalGeneration.py"
      }
    ]
  },
  "image-text-to-text": {
    filePaths: [
      // default model loader
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/model_loader.py`,
        to: "model_loader.py"
      },
      // endpoint handler
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/templates/multimodal/image-text-to-text/containerFiles/run_endpoint_handler.py`,
        to: "run_endpoint_handler.py"
      },
      // task specific model_entity.py
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/model_entity.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/model_entity.py"
      },
      // specific architecture updates
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/architectures/Gemma3ForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/architectures/Gemma3ForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/image_text_to_text/architectures/Gemma3ForConditionalGeneration.py"
      },
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/architectures/Idefics3ForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/architectures/Idefics3ForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/image_text_to_text/architectures/Idefics3ForConditionalGeneration.py"
      },
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/architectures/Llama4ForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/architectures/Llama4ForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/image_text_to_text/architectures/Llama4ForConditionalGeneration.py"
      },
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/architectures/MllamaForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/architectures/MllamaForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/image_text_to_text/architectures/MllamaForConditionalGeneration.py"
      },
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/architectures/PaliGemmaForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/architectures/PaliGemmaForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/image_text_to_text/architectures/PaliGemmaForConditionalGeneration.py"
      },
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/image_text_to_text/architectures/Qwen2VLForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/image_text_to_text/architectures/Qwen2VLForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/image_text_to_text/architectures/Qwen2VLForConditionalGeneration.py"
      }
    ]
  },
  "video-text-to-text": {
    filePaths: [
      // default model loader
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/model_loader.py`,
        to: "model_loader.py"
      },
      // endpoint handler
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/templates/multimodal/video-text-to-text/containerFiles/run_endpoint_handler.py`,
        to: "run_endpoint_handler.py"
      },
      // specific architecture updates
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/video_text_to_text/architectures/Idefics3ForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/video_text_to_text/architectures/Idefics3ForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/video_text_to_text/architectures/Idefics3ForConditionalGeneration.py"
      },
      {
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/template_extensions/custom_model_loader/architecture_registry/architecture_registry_module/tasks/video_text_to_text/architectures/LlavaNextVideoForConditionalGeneration.py`,
        to: "architecture_registry_module/tasks/video_text_to_text/architectures/LlavaNextVideoForConditionalGeneration.py",
        // only modify those which have this arch
        mustHaveFile:
          "architecture_registry_module/tasks/video_text_to_text/architectures/LlavaNextVideoForConditionalGeneration.py"
      }
    ]
  }
};

async function main() {
  const ROOT_DIR = pathModule.resolve(`${__dirname}/../../../modelsRepo`);

  const pathsToIterateOver = tasksToUpdate.map(task => `${ROOT_DIR}/${task}`);

  const modelPathObjects = await higherOrderIterator(
    pathsToIterateOver,
    undefined,
    undefined,
    ROOT_DIR
  );

  const modelsToUpdate = [];

  for (const modelPathObject of modelPathObjects) {
    // using an obj allows us to overrride existing paths if task specific files are specified
    const _filesToUpdate = {};
    const { task, modelId, filePath } = modelPathObject;

    // skip any tasks you don't want to update
    if (!tasksToUpdate.includes(task)) {
      continue;
    }

    // update root level files
    for (const file of filesToUpdate) {
      _filesToUpdate[file] = {
        name: file,
        from: `${BYTEZ_API_UTILITIES_PATH}/jobRunner/templates/default/${file}`,
        to: `${filePath}/${file}`
      };
    }

    // if there are task specific files, add those
    const config = filesToUpdateForSpecificTasks[task];

    if (config) {
      for (const { from, to: name, mustHaveFile } of config.filePaths) {
        const toPath = `${ROOT_DIR}/${task}/${modelId}/${name}`;
        const fromPath = pathModule.resolve(from);
        if (mustHaveFile) {
          const exists = await checkExistence(toPath);

          if (exists) {
            // now update the specific file

            _filesToUpdate[name] = {
              name,
              from: fromPath,
              to: toPath
            };
          }
          continue;
        }

        _filesToUpdate[name] = {
          name,
          from: fromPath,
          to: toPath
        };
      }
    }

    modelsToUpdate.push({
      ...modelPathObject,
      filesToUpdate: Object.values(_filesToUpdate)
    });
  }

  const updatedModels = [];
  const notUpdatedModels = [];

  const failedModels = [];

  for (const [index, modelPathObject] of modelsToUpdate.map((v, i) => [i, v])) {
    {
      const { modelId, githubLink, file, filePath, task, filesToUpdate } =
        modelPathObject;

      console.log(
        `On model: ${modelId} (${index + 1}/${modelsToUpdate.length})`
      );

      let failed = false;
      let updated = false;

      for (const { name, from, to: fileToUpdatePath } of filesToUpdate) {
        // console.log("Updating file: ", name);

        // console.log(`${from} --> ${fileToUpdatePath}`);

        const newFileBuffer = await fs.readFile(from);

        const newFileContents = newFileBuffer.toString();

        // console.log("New file is:\n\n", newFileContents);

        // overwrite the target file with the new file contents
        try {
          const exists = await checkExistence(fileToUpdatePath);

          if (!exists) {
            await fs.writeFile(fileToUpdatePath, newFileBuffer);
            updated = true;

            continue;
          }

          const oldFileContents = await getFileString(fileToUpdatePath);

          if (oldFileContents !== newFileContents) {
            await fs.writeFile(fileToUpdatePath, newFileBuffer);
            updated = true;
          }
        } catch (error) {
          console.error(error);
          failed = true;
        }
      }
      if (updated) {
        updatedModels.push(modelPathObject);
      }

      if (!updated) {
        notUpdatedModels.push(modelPathObject);
      }

      if (failed) {
        failedModels.push(modelPathObject);
      }
    }
  }

  console.log(`Total models: ${modelPathObjects.length}`);
  console.log(`${updatedModels.length} models were updated`);
  console.log(
    `${notUpdatedModels.length} models had the same contents and were not updated`
  );

  console.log("Number of models that failed: ", failedModels.length);

  debugger;
}

async function checkExistence(path) {
  try {
    await fs.stat(path);
    return true;
  } catch (error) {
    // console.error(error);
    return false;
  }
}

async function copy(from, to) {
  const string = await getFileString(from);
  await fs.writeFile(to, string);
}

async function getFileString(path) {
  const buffer = await fs.readFile(path);
  const string = buffer.toString();

  return string;
}

if (require.main === module) {
  main();
}
