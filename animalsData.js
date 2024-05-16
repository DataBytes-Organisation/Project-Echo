const animals = [
  {
    name: "Kangaroo",
    src: "/images/1.png",
    count: 45,
    description:
      "Kangaroos are large marsupials that are native to Australia. They have powerful hind legs and large feet for jumping and a strong tail for balance while moving.",
  },
  {
    name: "Crane",
    src: "/images/2.png",
    count: 15,
    description:
      "Cranes are tall birds with long legs and necks. They are known for their graceful, elaborate dances during mating rituals.",
  },
  {
    name: "Rabbit",
    src: "/images/3.png",
    count: 200,
    description:
      "Rabbits are small mammals with fluffy, short tails, whiskers and distinctive long ears. They are often found in meadows, woods, forests, grasslands, deserts, and wetlands.",
  },
  {
    name: "Groundhog",
    src: "/images/4.png",
    count: 7,
    description:
      "Groundhogs, also known as woodchucks, are large ground squirrels known for their burrowing habits and are native to North America.",
  },
  {
    name: "Dove",
    src: "/images/5.png",
    count: 300,
    description:
      "Doves are birds that are known for their soft cooing sounds and are often associated with peace and love.",
  },
  {
    name: "Dove",
    src: "/images/6.png",
    count: 300,
    description:
      "Doves are birds that are known for their soft cooing sounds and are often associated with peace and love.",
  },
  {
    name: "Chameleon",
    src: "/images/7.png",
    count: 180,
    description:
      "Chameleons are distinctive and highly specialized lizards known for their ability to change their color, their extrudable tongues, and their eyes which can pivot and focus independently.",
  },
  {
    name: "Sheep",
    src: "/images/8.png",
    count: 1000,
    description:
      "Sheep are domesticated animals kept as livestock. They are raised for their wool, meat, and milk.",
  },
  {
    name: "Frog",
    src: "/images/9.png",
    count: 4800,
    description:
      "Frogs are amphibians characterized by their short bodies, webbed fingers and toes, protruding eyes, and the ability to leap.",
  },
  {
    name: "Snail",
    src: "/images/10.png",
    count: 3000,
    description:
      "Snails are gastropod mollusks with a coiled shell large enough to retreat into. They are known for their slow movement.",
  },
  {
    name: "Spider",
    src: "/images/11.png",
    count: 4500,
    description:
      "Spiders are arachnids known for their eight legs and the ability to spin intricate webs for trapping their prey.",
  },
  {
    name: "Cnidarians",
    src: "/images/12.png",
    count: 330,
    description:
      "Cnidarians include jellyfish, corals, and sea anemones, characterized by their stinging cells.",
  },
];

const totalAnimals = animals.reduce((acc, animal) => acc + animal.count, 0);

function redirectToConservation(animalName, count, description) {
  const percentage = ((count / totalAnimals) * 100).toFixed(2);
  window.location.href = `animalPercent.html?animal=${encodeURIComponent(
    animalName
  )}&percentage=${percentage}&description=${encodeURIComponent(description)}`;
}
