const fs = require("fs").promises;

const path = require("path");

const [rootDir] = path.resolve(`${__dirname}/../`).split("/").slice(-1);

async function higherOrderIterator(
  pathsToIterateOver,
  callback = async (index, modelPathObject, modelPathObjects) => undefined,
  ignoreList = ["_repoManagement"]
) {
  if (pathsToIterateOver.constructor.name === "String") {
    pathsToIterateOver = [pathsToIterateOver];
  }

  const modelPathObjects = [];

  for (const pathToIterateOver of pathsToIterateOver) {
    const resolvedPath = path.resolve(pathToIterateOver);

    console.log(`Retrieving files from: ${resolvedPath}`);

    const files = await fs.readdir(resolvedPath, {
      recursive: true
    });

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

      const fullFilePath = path.resolve(`${pathToIterateOver}/${file}`);

      const [_, relativePath] = fullFilePath.split(`${rootDir}/`);

      const parts = relativePath.split("/");
      // const parts = file.split("/");

      if (parts.length === 3 && !file.includes(".git")) {
        const [task, org, model] = parts;

        const filePath = `${resolvedPath}/${file}`;

        modelPathObjects.push({
          task,
          modelId: `${org}/${model}`,
          githubLink: `https://github.com/Bytez-com/models/tree/main/${task}/${org}/${model}`,
          file,
          filePath
        });
      }
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
