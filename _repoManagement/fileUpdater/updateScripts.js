const { higherOrderIterator } = require("../higherOrderIterator");

const fs = require("fs").promises;

async function main() {
  const nameOfFileToUpdate = "loading_tracker.py";

  const newFilePath = `${__dirname}/../../../templates/default/${nameOfFileToUpdate}`;

  const newFileBuffer = await fs.readFile(newFilePath);

  const newFileContents = newFileBuffer.toString();

  console.log("New file is:\n\n", newFileContents);

  const updatedModels = [];
  const notUpdatedModels = [];

  const failedModels = [];

  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const modelPathObjects = await higherOrderIterator(
    pathToIterateOver,
    async (index, modelPathObject, modelPathObjects) => {
      const { modelId, githubLink, file, filePath } = modelPathObject;

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

          return;
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
  );

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

if (require.main === module) {
  main();
}
