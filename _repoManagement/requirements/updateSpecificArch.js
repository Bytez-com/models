const fs = require("fs").promises;
const path = require("path");

const { cypherQuery } = require("../../../../bytez-api-utilities/neo4j");

async function getModelsForArch(architecture) {
  const models = await cypherQuery({
    cypher: `
    MATCH (model:Model)--(arch:Architecture{name: "${architecture}"})
    RETURN model.name as modelId
    `
  });

  return models;
}

async function main() {
  try {
    const task = "image-text-to-text";
    const formattedTask = task.replaceAll("-", "_");

    const architectures = [
      "Qwen2VLForConditionalGeneration",
      "Idefics3ForConditionalGeneration",
      "PaliGemmaForConditionalGeneration"
    ];

    const results = [];

    for (const architecture of architectures) {
      let updated = 0;
      let same = 0;
      const doesNotExistList = [];

      const models = await getModelsForArch(architecture);

      console.log(`Number of models for ${architecture}, ${models.length}`);

      for (const { modelId } of models) {
        const file = `architecture_registry_module/tasks/${formattedTask}/architectures/${architecture}.py`;

        const templatePath = path.resolve(
          `${__dirname}/../../../template_extensions/custom_model_loader/architecture_registry/${file}`
        );

        const modelDir = path.resolve(`${__dirname}/../../${task}/${modelId}`);

        const filePath = `${modelDir}/${file}`;

        try {
          const buffer = await fs.readFile(templatePath);

          const newString = buffer.toString();

          const oldBuffer = await fs.readFile(filePath);

          const oldString = oldBuffer.toString();

          await fs.writeFile(filePath, newString);

          if (oldString !== newString) {
            updated++;
            continue;
          } else {
            same++;
          }
        } catch (error) {
          await fs
            .rmdir(modelDir, { force: true, recursive: true })
            .then(() => console.log(`Removed: ${modelDir}`))
            .catch(console.error);

          console.error(error);

          doesNotExistList.push(modelId);
        }
      }

      results.push({
        architecture,
        updated,
        same,
        total: models.length,
        doesNotExistList
      });
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

      const a = 2;
    }

    const a = 2;
  } catch (error) {
    console.error(error);

    const a = 2;
  }
}

main();
