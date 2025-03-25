const fs = require("fs").promises;

const fileExists = path =>
  fs
    .stat(path)
    .then(() => true)
    .catch(() => false);

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

module.exports = { fileExists, requirementsAsSet };
