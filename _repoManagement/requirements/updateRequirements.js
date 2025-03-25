/* eslint-disable max-len */
const fs = require("fs").promises;

const { higherOrderIterator } = require("../higherOrderIterator");
const { requirementsAsSet, fileExists } = require("./utils");

// this is a map that includes dependencies if they are not already there
const CO_DEPENDENCIES_MAP = {
  // we need to downgrade the packaging and setup tools in order to use "transformers-stream-generator" for models that use it
  "transformers-stream-generator": ["setuptools==68.2.2"]
};

async function main() {
  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const modelsUpdated = [];
  const modelsNotUpdated = [];

  await higherOrderIterator(
    pathToIterateOver,
    async (index, modelPathObject, modelPathObjects) => {
      const { modelId, filePath } = modelPathObject;

      // if (modelId !== "Qwen/Qwen-7B-Chat") {
      //   return;
      // }

      const requirementsPath = `${filePath}/requirements.txt`;

      console.log(
        `On model: ${requirementsPath} (${index + 1}/${
          modelPathObjects.length
        })`
      );

      const exists = await fileExists(requirementsPath);

      if (!exists) {
        return;
      }

      // get our requirements for what's on github
      const [requirements, requirementsNameMap, requirementsString] =
        await requirementsAsSet(requirementsPath);

      for (const name of Object.keys(requirementsNameMap)) {
        const codeps = CO_DEPENDENCIES_MAP[name];

        if (codeps) {
          for (const dep of codeps) {
            const [name] = dep.split("==");

            requirementsNameMap[name] = dep;
          }
        }
      }

      const updatedRequirements = Object.values(requirementsNameMap);

      const newRequirementsString = updatedRequirements.join("\n");

      if (requirementsString === newRequirementsString) {
        modelsNotUpdated.push(modelPathObject);
        return;
      }

      updatedRequirements.sort();

      // console.log(`Old requirements are:\n${requirementsString}`);
      // console.log(`New requirements are:\n${newRequirementsString}`);

      await fs.writeFile(requirementsPath, newRequirementsString);

      modelsUpdated.push(modelPathObject);
    }
  );

  console.log(`Models updated: `, modelsUpdated);
  console.log(`Models not updated: `, modelsNotUpdated);

  debugger;
}

if (require.main === module) {
  main();
}
