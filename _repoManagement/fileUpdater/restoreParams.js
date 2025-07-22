// last good commit

// 59fd477db400148e3a4d77af14f57e5627fae6a5

// 59fd477db400148e3a4d77af14f57e5627fae6a5

// we need to, in a copy of the models repo, go to one commit before the commit above and scan each model_loader.py file for the changed tasks, and get a list of the special params that are added to each model.

// We then need to update the params in the new model_loader.py

// NOTE, it's likely that trust_remote_code can always be set to True without consequence

// THIS IS *THE* COMMIT

// 3b50e310103d7711f3a6324bba60f12edcdb4a3e

const COMMIT = "3b50e310103d7711f3a6324bba60f12edcdb4a3e";

const { higherOrderIterator } = require("../higherOrderIterator");

const fs = require("fs").promises;

const TASKS_TO_UPDATE = [
  "summarization",
  "translation",
  "text-generation",
  "text2text-generation"
  // "image-text-to-text",
  // "video-text-to-text",
  // "audio-text-to-text"
];

const ALL_KWARGS = {};

const BAD_MODELS = {};

async function main() {
  const rootDir = `${__dirname}/../../../modelsRepoOld`;

  const pathsToIterateOver = TASKS_TO_UPDATE.map(task => `${rootDir}/${task}`);

  const modelPathObjectsOld = await higherOrderIterator(
    pathsToIterateOver,
    undefined,
    undefined,
    rootDir
  );
  // const modelPathObjects = await getModelObjects(
  //   `${__dirname}/../../../modelsRepo`
  // );

  const targetFile = "model_loader.py";

  for (const [index, modelPathObject] of modelPathObjectsOld.map((v, i) => [
    i,
    v
  ])) {
    {
      const { modelId, filePath, task } = modelPathObject;

      const githubUrl = `https://github.com/Bytez-com/models/blob/${COMMIT}/${task}/${modelId}/${targetFile}`;

      console.log(
        `On model: ${modelId} (${index + 1}/${modelPathObjectsOld.length})`
      );
      const oldFilePath = `${filePath}/${targetFile}`;

      // overwrite the target file with the new file contents

      try {
        const buffer = await fs.readFile(oldFilePath);
        var oldFileContents = buffer.toString();

        try {
          var args = pythonArgsToJs(oldFileContents);
        } catch (error) {
          // this is to handle anything that has the old model iterator
          args = pythonArgsToJsOld(oldFileContents);

          delete args.device_map;

          const keys = Object.keys(args);

          if (keys.length === 0) {
            continue;
          }

          for (const key of keys) {
            ALL_KWARGS[key] = true;
          }

          args = { task: "TASK", model: "MODEL_ID", ...args };
        }

        if (!args.task) {
          throw new Error("No task found, this should never happen");
        }

        if (!args.model) {
          throw new Error("No model found, this should never happen");
        }

        const keys = Object.keys(args);

        for (const key of keys) {
          ALL_KWARGS[key] = true;
        }

        // has extra params
        if (keys.length > 2) {
          const fileToUpdatePath = oldFilePath.replace(
            "modelsRepoOld",
            "modelsRepo"
          );

          const fileToUpdateBuffer = await fs.readFile(fileToUpdatePath);
          const fileToUpdateContents = fileToUpdateBuffer.toString();

          var newModelLoader = updateParams(fileToUpdateContents, args);

          await fs.writeFile(fileToUpdatePath, newModelLoader, "utf8");
        }

        if (keys.length < 1) {
          throw new Error(
            "No pipeline() params found, this should never happen"
          );
        }
      } catch (error) {
        BAD_MODELS[githubUrl] = true;

        // console.error(error);
        // console.log(githubUrl);

        // const a = 2;
      }
    }
  }

  const a = 2;
}

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    const a = 2;
  });
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

function pythonArgsToJsOld(modelLoader) {
  const [, paramsMatch = ""] =
    modelLoader.match(/### params ###([\s\S]*?)\n\s*\)/) || [];
  const paramsArray = paramsMatch
    .trim()
    .split("\n")
    .map(line => line.trim())
    .filter(line => line && line.includes("="))
    .map(line => {
      const [key, value] = line.split("=");
      return [key.trim(), value.trim()];
    });

  const params = {};

  for (let [key, value] of paramsArray) {
    if (value.startsWith("{") && value.endsWith("}")) {
      try {
        const dictionary = JSON.parse(value.toLowerCase());

        for (const param in dictionary) {
          dictionary[param] = dictionary[param] === true ? "True" : "False";
        }

        params[key] = dictionary;
      } catch (e) {
        params[key] = value; // fallback if JSON parsing fails
      }
    } else {
      params[key] = value.endsWith(",") ? value.slice(0, -1) : value;
    }
  }

  return params;
}

function updateParams(modelLoader, params) {
  const paramsString = createParamsString(params);
  const replaceParams = ({ modelLoader, identifier, paramsString }) =>
    modelLoader.replace(
      // this finds things like ### params ### or ### autoconfig params ###
      RegExp(`(### ${identifier} ###\\n)[\\s\\S]*?(?=\\n\\S)`),
      `$1${paramsString}`
    );
  let newModelLoader = replaceParams({
    modelLoader,
    identifier: "params",
    paramsString
  });

  return newModelLoader;
}
function createParamsString(params) {
  const paramsString = Object.entries(params)
    .map(([param, paramValue]) => {
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
    })
    .join(",\n");

  return paramsString;
}
