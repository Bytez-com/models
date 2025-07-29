const fs = require("fs").promises;

const { higherOrderIterator } = require("../higherOrderIterator");
const { requirementsAsSet } = require("./utils");

const AFFLICTED_MODELS = require("./afflictedModels.json");

const ADD_REQUIREMENTS = {
  transformers: "transformers==4.49.0"
};

const tasksToUpdate = ["image-text-to-text"];

const TORCH_VERSIONS = {};

async function main() {
  const modelsUpdated = [];

  const modelsWithNoRequirements = [];
  const modelsWithNoRequirementsFile = [];

  const modelsNotUpdated = [];

  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const pathsToIterateOver = tasksToUpdate.map(
    task => `${pathToIterateOver}/${task}`
  );

  const modelPathObjects = await higherOrderIterator(pathsToIterateOver);

  for (const [index, modelPathObject] of modelPathObjects.map((v, i) => [
    i,
    v
  ])) {
    const { task, modelId, filePath } = modelPathObject;

    if (!AFFLICTED_MODELS.includes(modelId)) {
      continue;
    }

    // if (modelId !== "zuxyfox/baloon_detr_freeze") {
    //   return;
    // }

    const requirementsPath = `${filePath}/requirements.txt`;

    console.log(
      `On model: ${requirementsPath} (${index + 1}/${modelPathObjects.length})`
    );

    const exists = await fs
      .stat(requirementsPath)
      .then(() => true)
      .catch(() => false);

    if (!exists) {
      modelsWithNoRequirementsFile.push(modelPathObject);
      continue;
    }

    const [requirements, requirementsNameMap, requirementsString] =
      await requirementsAsSet(requirementsPath);

    // we only care about what the original requirements have that our instance's requirements don't have
    const updatedRequirements = [];

    for (const reqName of [...requirements.values()]) {
      const fullReq = requirementsNameMap[reqName];

      const version = ADD_REQUIREMENTS[reqName];

      if (version === fullReq) {
        updatedRequirements.push(version);
        continue;
      }

      updatedRequirements.push("transformers==4.49.0");
    }

    const newRequirementsString = updatedRequirements.join("\n");

    if (requirementsString === newRequirementsString) {
      modelsNotUpdated.push(modelPathObject);
      continue;
    }

    console.log(`New requirements are:\n${newRequirementsString}`);

    await fs.writeFile(requirementsPath, newRequirementsString);

    modelsUpdated.push(modelPathObject);
  }

  console.log(`Models updated: `, modelsUpdated.length);
  console.log(`Models not updated: `, modelsNotUpdated.length);
  console.log(`Models with no reqs: `, modelsWithNoRequirements.length);
  console.log(
    `Models with no req files: `,
    modelsWithNoRequirementsFile.length
  );

  console.log(
    `The following packages were found across the repo that are not in the base ami: `,
    Object.keys(TORCH_VERSIONS).sort()
  );

  debugger;
}

if (require.main === module) {
  main();
}
