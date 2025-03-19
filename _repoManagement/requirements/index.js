const { higherOrderIterator } = require("../higherOrderIterator");

const fs = require("fs").promises;

const TORCH_VERSION = "2.6.0";

const LOCKED_NAME_TO_FULLNAME_MAP = {
  torchaudio: `torchaudio==${TORCH_VERSION}`,
  // this maps to torch 2.6.0
  torchvision: `torchvision==0.21`
};

const ALL_PACKAGES_NOT_INSTALLED_ON_INSTANCE = {};

async function main() {
  const instanceRequirementsPath = `${__dirname}/instanceRequirements.txt`;

  const [instanceRequirements] = await requirementsAsSet(
    instanceRequirementsPath
  );

  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const modelsUpdated = [];

  const modelsWithNoRequirements = [];
  const modelsWithNoRequirementsFile = [];

  const modelsNotUpdated = [];

  await higherOrderIterator(
    pathToIterateOver,
    async (index, modelPathObject, modelPathObjects) => {
      const { filePath } = modelPathObject;

      const requirementsPath = `${filePath}/requirements.txt`;

      console.log(
        `On model: ${requirementsPath} (${index + 1}/${
          modelPathObjects.length
        })`
      );

      const exists = await fs
        .stat(requirementsPath)
        .then(() => true)
        .catch(() => false);

      if (!exists) {
        modelsWithNoRequirementsFile.push(modelPathObject);
        return;
      }

      const [requirements, requirementsNameMap, requirementsString] =
        await requirementsAsSet(requirementsPath);

      // get what's in the original requirements that's not in our instance's requirements
      const difference = requirements.difference(instanceRequirements);

      const differenceReqNames = [...difference.values()];

      if (differenceReqNames.length === 0) {
        modelsWithNoRequirements.push(modelPathObject);
        return;
      }

      // we only care about what the original requirements have that our instance's requirements don't have
      const updatedRequirements = [];

      for (const reqName of differenceReqNames) {
        const fullReq = requirementsNameMap[reqName];

        ALL_PACKAGES_NOT_INSTALLED_ON_INSTANCE[fullReq] ??= [];
        ALL_PACKAGES_NOT_INSTALLED_ON_INSTANCE[fullReq].push(
          modelPathObject.file
        );

        const lockedRequirement = LOCKED_NAME_TO_FULLNAME_MAP[reqName];

        if (lockedRequirement) {
          updatedRequirements.push(lockedRequirement);
          continue;
        }

        updatedRequirements.push(fullReq);
      }

      const newRequirementsString = updatedRequirements.join("\n");

      if (requirementsString === newRequirementsString) {
        modelsNotUpdated.push(modelPathObject);
        return;
      }

      // console.log(`New requirements are:\n${newRequirementsString}`);

      await fs.writeFile(requirementsPath, newRequirementsString);

      modelsUpdated.push(modelPathObject);
    }
  );

  console.log(`Models updated: `, modelsUpdated);
  console.log(`Models not updated: `, modelsNotUpdated);
  console.log(`Models with no reqs: `, modelsWithNoRequirements);

  console.log(
    `The following packages were found across the repo that are not in the base ami: `,
    Object.keys(ALL_PACKAGES_NOT_INSTALLED_ON_INSTANCE).sort()
  );

  debugger;
}

async function requirementsAsSet(path) {
  const buffer = await fs.readFile(path);

  const string = buffer.toString().trim();

  const nameToFullLineMap = {};

  const array = string.split("\n").map(item => {
    const [reqName] = item.split("==");

    nameToFullLineMap[reqName] = item;

    return reqName;
  });

  const set = new Set(array);

  return [set, nameToFullLineMap, string];
}

if (require.main === module) {
  main();
}
