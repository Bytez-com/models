const fs = require("fs").promises;

const { higherOrderIterator } = require("../higherOrderIterator");
const { requirementsAsSet } = require("./utils");

const REMOVE_REQUIREMENTS = {
  transformers: "transformers==4.44.0"
};

const tasksToUpdate = [
  // "summarization",
  // "translation",
  "text-generation"
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
  // "zero-shot-image-classification",
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

    if (requirementsString === "") {
      continue;
    }

    // we only care about what the original requirements have that our instance's requirements don't have
    const updatedRequirements = [];

    for (const reqName of [...requirements.values()]) {
      const fullReq = requirementsNameMap[reqName];

      const version = REMOVE_REQUIREMENTS[reqName];

      if (version === fullReq) {
        continue;
      }

      updatedRequirements.push(fullReq);
    }

    const newRequirementsString = updatedRequirements.join("\n");

    if (requirementsString === newRequirementsString) {
      modelsNotUpdated.push(modelPathObject);
      continue;
    }

    console.log(`New requirements are:\n${newRequirementsString}`);

    // await fs.writeFile(requirementsPath, newRequirementsString);

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
