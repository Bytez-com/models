const { higherOrderIterator } = require("../higherOrderIterator");

const fs = require("fs").promises;

const tasksToUpdate = [
  // "summarization",
  // "translation",
  "text-generation",
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
  "image-text-to-text",
  "audio-text-to-text",
  "video-text-to-text"
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
  // "text-to-speech",
  // "text-to-audio",
  // "automatic-speech-recognition",
  // "audio-classification"
];

const filesToUpdate = [
  // "model.py"
  // "download_bytez_repo.py",
  // "environment.py"
  // "model_loader.py",
  // // "streamer.py"
  // // "utils.py",
  "vllm_loader.py"
  // "vllm_server.py"
  // "vllm_mocks.py"
];

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
        const { modelId, githubLink, file, filePath, task } = modelPathObject;

        if (!tasksToUpdate.includes(task)) {
          continue;
        }

        console.log(
          `On model: ${modelId} (${index + 1}/${modelPathObjects.length})`
        );

        const fileToUpdatePath = `${filePath}/${nameOfFileToUpdate}`;

        // overwrite the target file with the new file contents
        try {
          const exists = await fs
            .stat(fileToUpdatePath)
            .then(() => true)
            .catch(() => false);

          if (!exists) {
            await fs.writeFile(fileToUpdatePath, newFileBuffer);
            updatedModels.push(modelPathObject);

            continue;
          }

          const buffer = await fs.readFile(fileToUpdatePath);
          const oldFileContents = buffer.toString();

          if (oldFileContents !== newFileContents) {
            await fs.writeFile(fileToUpdatePath, newFileBuffer);
            updatedModels.push(modelPathObject);
          } else {
            notUpdatedModels.push(modelPathObject);
          }
        } catch (error) {
          failedModels.push(modelPathObject);
        }
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
