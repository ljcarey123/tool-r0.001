// Loads the generated JSON in docs/data (see scripts/build_explorer_data.py).
// Each file is fetched once; sections ask for what they need.

const cache = new Map();

export function load(name) {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(`data/${name}.json`).then((res) => {
        if (!res.ok) throw new Error(`data/${name}.json: ${res.status} ${res.statusText}`);
        return res.json();
      }),
    );
  }
  return cache.get(name);
}
