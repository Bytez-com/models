const fs = require("fs").promises;
const path = require("path");

const { cypherQuery } = require("../../../../bytez-api-utilities/neo4j");

const MISSING_MODELS = {};

// NOTE these do need to be updated to use the most recent architectures, which have to be enumerated manually (for now)
// e.g.
// MATCH (task:Tag)--(model:Model)--(arch:Architecture)
// WHERE task.name = "video-text-to-text"
// RETURN DISTINCT arch.name
// ORDER BY arch.name ASC

const updates = [
  {
    task: "audio-text-to-text",
    files: [
      `architecture_registry_module/tasks/audio_text_to_text/model_entity.py`
    ],
    architectures: [
      {
        //
        name: "Qwen2AudioForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/audio_text_to_text/architectures/Qwen2AudioForConditionalGeneration.py"
        ]
      }
    ]
  },
  {
    task: "image-text-to-text",
    files: [
      `architecture_registry_module/tasks/image_text_to_text/model_entity.py`
    ],
    architectures: [
      {
        name: "Gemma3ForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/image_text_to_text/architectures/Gemma3ForConditionalGeneration.py"
        ]
      },
      {
        name: "Idefics3ForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/image_text_to_text/architectures/Idefics3ForConditionalGeneration.py"
        ]
      },
      {
        name: "Llama4ForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/image_text_to_text/architectures/Llama4ForConditionalGeneration.py"
        ]
      },
      {
        name: "MllamaForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/image_text_to_text/architectures/MllamaForConditionalGeneration.py"
        ]
      },
      {
        name: "PaliGemmaForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/image_text_to_text/architectures/PaliGemmaForConditionalGeneration.py"
        ]
      },
      {
        name: "Qwen2VLForConditionalGeneration",
        files: [
          "architecture_registry_module/tasks/image_text_to_text/architectures/Qwen2VLForConditionalGeneration.py"
        ]
      },
      {
        // these don't have a special class, they use the generic
        name: "Qwen2_5_VLForConditionalGeneration",
        files: []
      }
    ]
  },
  {
    task: "video-text-to-text",
    files: [
      `architecture_registry_module/tasks/video_text_to_text/model_entity.py`
    ],
    architectures: [
      {
        // these don't have a special class, they use the generic
        name: "LlavaNextVideoForConditionalGeneration",
        files: []
      }
    ]
  }
];

async function main() {
  try {
    const results = [];

    for (const { task, files: generalFiles, architectures } of updates) {
      for (const {
        name: architectureName,
        files: architectureFiles
      } of architectures) {
        let updated = 0;
        let same = 0;
        const doesNotExistList = [];

        const models = await getModelsForArch(architectureName);

        console.log(
          `Number of models for ${architectureName}, ${models.length}`
        );

        for (const { modelId } of models) {
          for (const file of generalFiles) {
            const { fileChanged, exists } = await updateFile({
              task,
              file,
              modelId
            });

            if (fileChanged) {
              updated++;
            } else {
              same++;
            }

            if (!exists) {
              doesNotExistList.push(modelId);
            }
          }

          for (const file of architectureFiles) {
            const { fileChanged, exists } = await updateFile({
              task,
              file,
              modelId
            });

            if (fileChanged) {
              updated++;
            } else {
              same++;
            }

            if (!exists) {
              doesNotExistList.push(modelId);
            }
          }
        }

        results.push({
          architecture: architectureName,
          updated,
          same,
          total: models.length,
          doesNotExistList
        });
      }
    }

    for (const {
      architecture,
      updated,
      same,
      total,
      doesNotExistList
    } of results) {
      console.log(
        `Architecture: ${architecture}, total: ${total}, does not exist: ${doesNotExistList.length}, upated: ${updated}, no op ${same}`
      );
    }
  } catch (error) {
    console.error(error);

    debugger;
  }

  const missingModelIds = Object.keys(MISSING_MODELS);
  const missingModelValues = Object.values(MISSING_MODELS);

  console.log(
    `Models that exist in the graph, but not in the repo's code: ${missingModelIds.length}`
  );

  debugger;
}

async function getModelsForArch(architecture) {
  const models = await cypherQuery({
    cypher: `
    MATCH (model:Model)--(arch:Architecture{name: "${architecture}"})
    RETURN model.name as modelId
    `
  });

  return models;
}

async function updateFile({ task, modelId, file }) {
  const templatePath = path.resolve(
    `${__dirname}/../../../../bytez-api-utilities/jobRunner/template_extensions/custom_model_loader/architecture_registry/${file}`
  );

  const pathToModelDirInRepo = path.resolve(
    `${__dirname}/../../${task}/${modelId}`
  );

  const pathToModelInRepo = path.resolve(`${pathToModelDirInRepo}/${file}`);

  try {
    const buffer = await fs.readFile(templatePath);

    const newString = buffer.toString();

    const oldBuffer = await fs.readFile(pathToModelInRepo);

    const oldString = oldBuffer.toString();

    await fs.writeFile(pathToModelInRepo, newString);

    const fileChanged = oldString !== newString;

    return { fileChanged, exists: true };
  } catch (error) {
    console.error(error);

    MISSING_MODELS[modelId] = { task, modelId };

    // await fs
    //   .rmdir(pathToModelDirInRepo, { force: true, recursive: true })
    //   .then(() => console.log(`Removed: ${modelDir}`))
    //   .catch(console.error);

    return { fileChanged: null, exists: false };
  }
}

main();
