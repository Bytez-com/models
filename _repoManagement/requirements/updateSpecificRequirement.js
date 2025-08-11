const { higherOrderIterator } = require("../higherOrderIterator");

const fs = require("fs").promises;

const { requirementsAsSet, fileExists } = require("./utils");

const tasksToUpdate = [
  // "summarization",
  // "translation",
  // "text-generation"
  // "text2text-generation",
  // "image-text-to-text",
  // "video-text-to-text",
  // "audio-text-to-text"
  // "visual-question-answering",
  // "document-question-answering",
  // "depth-estimation",
  // "image-classification",
  // "object-detection",
  // "image-segmentation",
  // "text-to-image",
  // "image-to-text",
  // "image-text-to-text",
  // "audio-text-to-text",
  // "video-text-to-text",
  // "unconditional-image-generation",
  // "video-classification",
  // "text-to-video",
  // "zero-shot-image-classification"
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
  // "text-generation",
  // "text2text-generation",
  // "fill-mask",
  // "sentence-similarity",
  "text-to-speech"
  // "text-to-audio",
  // "automatic-speech-recognition",
  // "audio-classification"
];

const filesToUpdate = [
  "requirements.txt"
  // "download_bytez_repo.py",
  // "environment.py"
  // "model_loader.py",
  // // "streamer.py"
  // // "utils.py",
  // "vllm_loader.py"
  // "vllm_server.py"
  // "vllm_mocks.py"
];

const REQUIREMENTS_TO_UPDATE = {
  transformers: "transformers==4.49.0"
};

async function main() {
  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const pathsToIterateOver = tasksToUpdate.map(
    task => `${pathToIterateOver}/${task}`
  );

  const rootDir = `${__dirname}/../../../modelsRepo`;

  const modelPathObjects = await higherOrderIterator(
    pathsToIterateOver,
    undefined,
    undefined,
    rootDir
  );

  for (const nameOfFileToUpdate of filesToUpdate) {
    const newFilePath = `${__dirname}/../../../templates/default/${nameOfFileToUpdate}`;

    const newFileBuffer = await fs.readFile(newFilePath);

    const newFileContents = newFileBuffer.toString();

    console.log("New file is:\n\n", newFileContents);

    const updatedModels = [];
    const notUpdatedModels = [];

    const failedModels = [];

    for (const [index, modelPathObject] of modelPathObjects.map((v, i) => [
      i,
      v
    ])) {
      {
        const { modelId, filePath } = modelPathObject;

        console.log(
          `On model: ${modelId} (${index + 1}/${modelPathObjects.length})`
        );

        const requirementsPath = `${filePath}/requirements.txt`;

        console.log(
          `On model: ${requirementsPath} (${index + 1}/${
            modelPathObjects.length
          })`
        );

        const [requirementsSet, requirementsNameMap, requirementsString] =
          await requirementsAsSet(requirementsPath);

        for (const [reqName, fullReq] of Object.entries(
          REQUIREMENTS_TO_UPDATE
        )) {
          requirementsNameMap[reqName] = fullReq;
        }

        const updatedRequirements = Object.values(requirementsNameMap);

        updatedRequirements.sort();

        const newRequirementsString = updatedRequirements.join("\n");

        if (requirementsString === newRequirementsString) {
          notUpdatedModels.push(modelPathObject);
          continue;
        }

        // console.log(`Old requirements are:\n${requirementsString}`);
        console.log(`New requirements are:\n${newRequirementsString}`);

        await fs.writeFile(requirementsPath, newRequirementsString);

        updatedModels.push(modelPathObject);
      }
    }

    console.log(`Total models: ${modelPathObjects.length}`);
    console.log(
      `File: ${nameOfFileToUpdate} updated for ${updatedModels.length} models`
    );
    console.log(
      `${notUpdatedModels.length} models had the same contents and were not updated`
    );

    console.log("Number of models that failed: ", failedModels.length);

    debugger;
  }
}

if (require.main === module) {
  main();
}
