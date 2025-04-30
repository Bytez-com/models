/* eslint-disable max-len */
const { higherOrderIterator } = require("../higherOrderIterator");
const { fileExists } = require("../requirements/utils");

const fs = require("fs").promises;
const path = require("path");

const TASK_TO_CATEGORY_MAP = require("../../../constants/taskCategorization/maps/taskToCategoryMap.json");

const TASKS_UPDATED_MAP = {};

const NAME_OF_FILE_TO_UPDATE = "model_loader.py";

const ARCHITECTURE_REGISTRY_PATH = `${__dirname}/../../../template_extensions/custom_model_loader/architecture_registry/model_loader.py`;

const TASK_PATHS = {
  "audio-classification": `${__dirname}/../../../templates/audio/audio-classification/containerFiles/${NAME_OF_FILE_TO_UPDATE}`,
  "text-to-image": `${__dirname}/../../../templates/computer-vision/text-to-image/containerFiles/${NAME_OF_FILE_TO_UPDATE}`,
  "text-to-video": `${__dirname}/../../../templates/computer-vision/text-to-video/containerFiles/${NAME_OF_FILE_TO_UPDATE}`,
  "unconditional-image-generation": `${__dirname}/../../../templates/computer-vision/unconditional-image-generation/containerFiles/${NAME_OF_FILE_TO_UPDATE}`,
  "sentence-similarity": `${__dirname}/../../../templates/natural-language-processing/sentence-similarity/containerFiles/${NAME_OF_FILE_TO_UPDATE}`,
  "audio-text-to-text": ARCHITECTURE_REGISTRY_PATH,
  "image-text-to-text": ARCHITECTURE_REGISTRY_PATH,
  "video-text-to-text": ARCHITECTURE_REGISTRY_PATH
};

const DEFAULT_MODEL_LOADER_PATH = `${__dirname}/../../../templates/default/${NAME_OF_FILE_TO_UPDATE}`;

const FAILED_TASKS = {};

async function main() {
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

      const newFileContents = await getFileString(
        NAME_OF_FILE_TO_UPDATE,
        category,
        task
      );

      // console.log("New file is:\n\n", newFileContents);

      console.log(
        `On model: ${modelId} (${index + 1}/${modelPathObjects.length})`
      );

      const fileToUpdatePath = `${filePath}/${NAME_OF_FILE_TO_UPDATE}`;

      // overwrite the target file with the new file contents
      const githubLink = `https://github.com/Bytez-com/models/blob/main/${task}/${modelId}/model_loader.py`;
      try {
        const exists = await fileExists(fileToUpdatePath);

        if (!exists) {
          await fs.writeFile(fileToUpdatePath, newFileContents);

          newFileModels.push(modelPathObject);
          TASKS_UPDATED_MAP[task] ??= [];
          TASKS_UPDATED_MAP[task].push(modelPathObject);
          return;
        }

        const buffer = await fs.readFile(fileToUpdatePath);
        const oldFileContents = buffer.toString();

        // now we take the args from the old file and put them into the new one
        const oldArgs = pythonArgsToJs(oldFileContents);
        const newFileContentsWithArgs = updatePythonFileWithArgs(
          newFileContents,
          oldArgs
        );

        if (newFileContents !== newFileContentsWithArgs) {
          const a = 2;
        }

        if (
          oldFileContents !== newFileContents &&
          // if we've already overwritten the file on a previous run
          oldFileContents !== newFileContentsWithArgs
        ) {
          await fs.writeFile(fileToUpdatePath, newFileContentsWithArgs);

          updatedModels.push(modelPathObject);
          TASKS_UPDATED_MAP[task] ??= [];
          TASKS_UPDATED_MAP[task].push(modelPathObject);
        } else {
          notUpdatedModels.push(modelPathObject);
        }
      } catch (error) {
        FAILED_TASKS[task] = true;
        failedModels.push(modelPathObject);
      }
    }
  );

  console.log(`Total models: ${modelPathObjects.length}`);
  console.log(
    `File: ${NAME_OF_FILE_TO_UPDATE} updated for ${updatedModels.length} models`
  );
  console.log(
    `File: ${NAME_OF_FILE_TO_UPDATE} created for ${newFileModels.length} models`
  );
  console.log(
    `${notUpdatedModels.length} models had the same contents and were not updated`
  );

  console.log("Number of models that failed: ", failedModels.length);

  for (const [task, models] of Object.entries(TASKS_UPDATED_MAP)) {
    console.log(`Task: ${task} updated for ${models.length} models`);
  }

  await fs.writeFile(
    `${__dirname}/modelsThatNeedToBeRerun.json`,
    JSON.stringify(failedModels, null, 2)
  );

  debugger;
}

function pythonArgsToJs(modelLoader) {
  const [, paramsMatch = ""] =
    modelLoader.match(/### params ###([\s\S]*?)(?=\n\S)/) || [];
  const paramsArray = paramsMatch
    .trim()
    .split("\n")
    .map(line => (line === "" ? undefined : line.trim().split(":")))
    .filter(defined => defined)
    .map(([key, value]) => {
      // a dictionary is now used for params, so we need to normalize the keys and values to JS strings
      return [key.replaceAll(`"`, ""), value];
    });
  const params = {};

  for (let [key, value] of paramsArray) {
    value = value.trim();

    if (value.startsWith("{") && value.endsWith("}")) {
      const dictionary = JSON.parse(value.toLowerCase());

      for (const param in dictionary) {
        dictionary[param] = dictionary[param] === true ? "True" : "False";
      }

      var paramValue = dictionary;
    } else {
      paramValue = value.endsWith(",") ? value.slice(0, -1) : value;
    }

    params[key.trim()] = paramValue;
  }

  return params;
}

function updatePythonFileWithArgs(modelLoader, params) {
  const paramsString = createParamsString(params);
  const replaceParams = ({ modelLoader, identifier, paramsString }) =>
    modelLoader.replace(
      // this finds things like ### params ### or ### autoconfig params ###
      RegExp(`(### ${identifier} ###\\n)[\\s\\S]*?(?=\\n\\S)`),
      `$1${paramsString}`
    );

  const newModelLoader = replaceParams({
    modelLoader,
    identifier: "params",
    paramsString
  });

  return newModelLoader;
}
function createParamsString(params) {
  const paramsArray = Object.entries(params).map(([param, paramValue]) => {
    if (paramValue.constructor === String) {
      var valueString = paramValue;
    } else {
      valueString = "{ ";

      for (const param in paramValue) {
        valueString += `"${param}": ${paramValue[param]}, `;
      }

      valueString = valueString.slice(0, -2) + " }";
    }

    return `    "${param}": ${valueString}`;
  });

  const paramsString = paramsArray.join(",\n") + ",";

  return paramsString;
}

const PATH_CACHE = {};

async function getFileString(nameOfFileToUpdate, category, task) {
  const key = `${nameOfFileToUpdate}-${category}-${task}`;

  if (PATH_CACHE[key]) {
    return PATH_CACHE[key];
  }

  const modelLoaderPath = TASK_PATHS[task]
    ? TASK_PATHS[task]
    : DEFAULT_MODEL_LOADER_PATH;

  const finalFilePath = path.resolve(modelLoaderPath);

  const newFileBuffer = await fs.readFile(finalFilePath);

  const newFileContents = newFileBuffer.toString();

  PATH_CACHE[key] = newFileContents;

  return newFileContents;
}

if (require.main === module) {
  main();
}
