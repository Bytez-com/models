const fs = require("fs").promises;
const path = require("path");

const { higherOrderIterator } = require("../higherOrderIterator");

const TASK_TO_CATEGORY_MAP = require("../../../constants/taskCategorization/maps/taskToCategoryMap.json");
const { fileExists, requirementsAsSet } = require("./utils");

const SKIP_MAP = {
  // "transformers==4.49.0": true,
  // "pillow==11.1.0": true,
  // "torchvision==0.21": true
};

const KEEP_LOCKED_MAP = {
  "librosa==0.10.2.post1": true
  // remove after testing
  // librosa: true,
  // "pytesseract==0.3.10": true,
  // pytesseract: true,
  // "pillow==10.4.0": true,
  // pillow: true,
  // timm: true,
  // "timm==1.0.7": true,
  // "timm==1.0.8": true,
  // "sentence-transformers": true,
  // "scipy==1.14.0": true,
  // "diffusers==0.29.2": true,
  // "diffusers==0.30.0": true,
  // "diffusers==0.30.1": true,
  // diffusers: true,
  // "av==12.3.0": true,
  // "av==12.4.0": true,
  // "av==12.5.0": true,
  // "av==12.6.0": true,
  // av: true
};

async function main() {
  const pathToIterateOver = `${__dirname}/../../../modelsRepo`;

  const modelsUpdated = [];
  const modelsNotUpdated = [];
  const modelsWithNoRequirementsFile = [];

  await higherOrderIterator(
    pathToIterateOver,
    async (index, modelPathObject, modelPathObjects) => {
      const { modelId, task, filePath } = modelPathObject;

      // if (modelId !== "Qwen/Qwen2-Audio-7B-Instruct") {
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
        modelsWithNoRequirementsFile.push(modelPathObject);
        return;
      }

      // get our requirements for what's on github
      const [requirements, requirementsNameMap, requirementsString] =
        await requirementsAsSet(requirementsPath);

      const category = TASK_TO_CATEGORY_MAP[task];

      if (!category) {
        throw new Error(
          "Task missing from the category map, update the category map"
        );
      }

      // now we see if we're missing any of the base requirements
      const templateRequirementsPath = path.resolve(
        `${__dirname}/../../../templates/${category}/${task}/containerFiles/requirements.txt`
      );

      const hasTemplateRequirements = await fileExists(
        templateRequirementsPath
      );

      if (hasTemplateRequirements) {
        const [templateRequirements, templateRequirementsNameMap] =
          await requirementsAsSet(templateRequirementsPath);
        // we want to add the missing reqs that are part of the template, and we want to not change templates that are already there
        // e.g. pillow is now missing, transformers is now missing, etc
        for (const reqName of Object.keys(templateRequirementsNameMap)) {
          const fullTemplateRequirement = templateRequirementsNameMap[reqName];
          const currentTemplateRequirement = requirementsNameMap[reqName];

          // we won't want anything that was added by the job runner to be set to undefined
          if (fullTemplateRequirement === undefined) {
            continue;
          }

          const isDifferent =
            currentTemplateRequirement !== fullTemplateRequirement;

          const isLocked = KEEP_LOCKED_MAP[currentTemplateRequirement];

          const shouldSkip = SKIP_MAP[fullTemplateRequirement];

          if (isDifferent && !isLocked && !shouldSkip) {
            console.log(
              `Changed: ${currentTemplateRequirement} to ${fullTemplateRequirement}`
            );
            requirementsNameMap[reqName] = fullTemplateRequirement;
          } else {
            const a = 2;
          }
        }
      }

      const filteredReqs = {};

      for (const [reqName, value] of Object.entries(requirementsNameMap)) {
        if (
          reqName.includes(
            "## The following requirements were added by pip freeze:"
          )
        ) {
          continue;
        }

        filteredReqs[reqName] = value;
      }

      const updatedRequirements = Object.values(filteredReqs);

      // updatedRequirements.sort();

      const newRequirementsString = updatedRequirements.join("\n");

      if (requirementsString === newRequirementsString) {
        modelsNotUpdated.push(modelPathObject);
        return;
      }

      // console.log(`Old requirements are:\n${requirementsString}`);
      // console.log(`New requirements are:\n${newRequirementsString}`);

      await fs.writeFile(requirementsPath, newRequirementsString);

      modelsUpdated.push(modelPathObject);
    }
  );

  console.log(`Models updated: `, modelsUpdated);
  console.log(`Models not updated: `, modelsNotUpdated);
  console.log(`Models with no req files: `, modelsWithNoRequirementsFile);

  debugger;
}

if (require.main === module) {
  main();
}
