const { readdir, readFile, writeFile } = require("fs").promises;

async function main() {
  const files = await readdir(__dirname, {
    recursive: true
  });

  const modelPathObjects = [];

  for (const file of files) {
    const parts = file.split("/");

    if (parts.length === 3 && !file.includes(".git")) {
      const [task, org, model] = parts;

      const filePath = `${__dirname}/${file}`;

      modelPathObjects.push({
        model,
        githubLink: `https://github.com/Bytez-com/models/tree/main/${task}/${org}/${model}`,
        file,
        filePath
      });
    }
  }

  const nameOfFileToUpdate = "server.py";

  const newFilePath = `${__dirname}/../templates/default/${nameOfFileToUpdate}`;

  const newFileBuffer = await readFile(newFilePath);

  const newFileContents = newFileBuffer.toString();

  console.log("New file is:\n\n", newFileContents);

  const updatedModels = [];
  const notUpdatedModels = [];

  const failedModels = [];

  for (const [index, modelPathObject] of modelPathObjects.map((v, i) => [
    i,
    v
  ])) {
    const { model, githubLink, file, filePath } = modelPathObject;

    console.log(`On model: ${model} (${index + 1}/${modelPathObjects.length})`);

    const fileToUpdatePath = `${filePath}/${nameOfFileToUpdate}`;

    // overwrite the target file with the new file contents
    try {
      const buffer = await readFile(fileToUpdatePath);
      const oldFileContents = buffer.toString();

      if (oldFileContents !== newFileContents) {
        await writeFile(fileToUpdatePath, newFileBuffer);
        updatedModels.push(modelPathObject);
      } else {
        notUpdatedModels.push(modelPathObject);
      }
    } catch (error) {
      failedModels.push(modelPathObject);
    }
  }

  console.log(`Total models: ${modelPathObjects.length}`);
  console.log(
    `File: ${nameOfFileToUpdate} updated for ${updatedModels.length} models`
  );
  console.log(
    `${notUpdatedModels.length} models had the same contents and were not updated`
  );

  console.log("Number of models that failed: ", failedModels.length);

  const a = 2;
}

if (require.main === module) {
  main();
}
