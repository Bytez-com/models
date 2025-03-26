/* eslint-disable max-len */
const { higherOrderIterator } = require("../higherOrderIterator");
const { fileExists } = require("../requirements/utils");

const fs = require("fs").promises;
const path = require("path");

const TASK_TO_CATEGORY_MAP = require("../../../constants/taskCategorization/maps/taskToCategoryMap.json");

const TASKS_UPDATED_MAP = {};

async function main() {
  const nameOfFileToUpdate = "run_endpoint_handler.py";

  const updatedModels = [];
  const newFileModels = [];
  const notUpdatedModels = [];

  const failedModels = [];

  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const modelPathObjects = await higherOrderIterator(
    pathToIterateOver,
    async (index, modelPathObject, modelPathObjects) => {
      const { modelId, task, filePath } = modelPathObject;

      // if (modelId !== "facebook/opt-30b") {
      //   return;
      // }

      const category = TASK_TO_CATEGORY_MAP[task];

      if (!category) {
        throw new Error(
          "Task missing from the category map, update the category map"
        );
      }

      const finalFilePath = await getHandlerPath(
        nameOfFileToUpdate,
        category,
        task
      );

      if (!finalFilePath) {
        throw new Error(
          "Could not find an endpoint handler, including a default"
        );
      }

      const newFileBuffer = await fs.readFile(finalFilePath);

      const newFileContents = newFileBuffer.toString();

      // console.log("New file is:\n\n", newFileContents);

      console.log(
        `On model: ${modelId} (${index + 1}/${modelPathObjects.length})`
      );

      const fileToUpdatePath = `${filePath}/${nameOfFileToUpdate}`;

      // overwrite the target file with the new file contents
      try {
        const exists = await fileExists(fileToUpdatePath);

        if (!exists) {
          await fs.writeFile(fileToUpdatePath, newFileBuffer);

          newFileModels.push(modelPathObject);
          TASKS_UPDATED_MAP[task] ??= [];
          TASKS_UPDATED_MAP[task].push(modelPathObject);
          return;
        }

        const buffer = await fs.readFile(fileToUpdatePath);
        const oldFileContents = buffer.toString();

        if (oldFileContents !== newFileContents) {
          await fs.writeFile(fileToUpdatePath, newFileBuffer);

          updatedModels.push(modelPathObject);
          TASKS_UPDATED_MAP[task] ??= [];
          TASKS_UPDATED_MAP[task].push(modelPathObject);
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
    `File: ${nameOfFileToUpdate} created for ${newFileModels.length} models`
  );
  console.log(
    `${notUpdatedModels.length} models had the same contents and were not updated`
  );

  console.log("Number of models that failed: ", failedModels.length);

  for (const [task, models] of Object.entries(TASKS_UPDATED_MAP)) {
    console.log(`Task: ${task} updated for ${models.length} models`);
  }

  debugger;
}

const PATH_CACHE = {};

async function getHandlerPath(nameOfFileToUpdate, category, task) {
  const key = `${nameOfFileToUpdate}-${category}-${task}`;

  if (PATH_CACHE[key]) {
    return PATH_CACHE[key];
  }

  const _defaultEndpointHandlerPath = `${__dirname}/../../../templates/default/${nameOfFileToUpdate}`;

  const _endpointHandlerPath = `${__dirname}/../../../templates/${category}/${task}/containerFiles/${nameOfFileToUpdate}`;

  const _finalFilePath = (await fileExists(_endpointHandlerPath))
    ? _endpointHandlerPath
    : (await fileExists(_defaultEndpointHandlerPath))
    ? _defaultEndpointHandlerPath
    : null;

  const finalFilePath = path.resolve(_finalFilePath);

  PATH_CACHE[key] = finalFilePath;

  return finalFilePath;
}

if (require.main === module) {
  main();
}
