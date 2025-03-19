const fs = require("fs").promises;

const path = require("path");

async function higherOrderIterator(
  pathToIterateOver,
  callback = async (index, modelPathObject, modelPathObjects) => undefined,
  ignoreList = ["_repoManagement"]
) {
  const resolvedPath = path.resolve(pathToIterateOver);

  console.log(`Retrieving files from: ${resolvedPath}`);

  const files = await fs.readdir(resolvedPath, {
    recursive: true
  });

  const modelPathObjects = [];

  for (const file of files) {
    let shouldSkip = false;

    for (const ignoreGlob of ignoreList) {
      if (file.includes(ignoreGlob)) {
        shouldSkip = true;
      }
    }

    if (shouldSkip) {
      continue;
    }

    const parts = file.split("/");

    if (parts.length === 3 && !file.includes(".git")) {
      const [task, org, model] = parts;

      const filePath = `${resolvedPath}/${file}`;

      modelPathObjects.push({
        modelId: `${org}/${model}`,
        githubLink: `https://github.com/Bytez-com/models/tree/main/${task}/${org}/${model}`,
        file,
        filePath
      });
    }
  }

  modelPathObjects.sort((a, b) => (a.filePath < b.filePath ? -1 : 1));

  for (const [index, modelPathObject] of modelPathObjects.map((v, i) => [
    i,
    v
  ])) {
    await callback(index, modelPathObject, modelPathObjects);
  }

  return modelPathObjects;
}

module.exports = { higherOrderIterator };
