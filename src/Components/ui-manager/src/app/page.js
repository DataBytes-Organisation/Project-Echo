import {
  Box,
  Heading,
  Text,
  Image,
  Button,
  VStack,
  Flex,
} from "@chakra-ui/react";

export default function Home() {
  return (
    <Flex direction="column" align="center" justify="center" padding="20px">
      {/* Hero Section */}
      <Box
        as="section"
        width="100%"
        textAlign="center"
        padding="50px 20px"
        background="linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('/imgs/homebg.jpg')"
        backgroundSize="cover"
        backgroundPosition="center"
        color="white"
      >
        <Heading as="h1" size="2xl" mb="20px">
          Welcome to Project Echo
        </Heading>
        <Text fontSize="xl" mb="30px">
          A cutting-edge bioacoustics tool to discover, track, and monitor
          endangered species in their natural environment.
        </Text>
        <Button size="lg" colorScheme="teal">
          Learn More
        </Button>
      </Box>

      {/* Mission and Vision Section */}
      <Box
        as="section"
        width="100%"
        padding="50px 20px"
        backgroundColor="white"
      >
        <VStack spacing="20px" maxW="800px" mx="auto">
          <Heading as="h2" size="xl">
            Our Mission
          </Heading>
          <Text fontSize="lg">
            At Project Echo, our mission is to provide a bioacoustics suite of
            tools through AI/ML end-to-end solutions to understand animal
            vocalizations, helping conservationists make informed decisions to
            protect endangered species.
          </Text>
        </VStack>
      </Box>

      {/* Key Features Section */}
      <Box
        as="section"
        width="100%"
        padding="50px 20px"
        backgroundColor="gray.100"
      >
        <VStack spacing="30px" maxW="800px" mx="auto">
          <Heading as="h2" size="xl">
            Key Features
          </Heading>
          <Text fontSize="lg">
            Explore the powerful features of Project Echo:
          </Text>
          <Flex justify="space-around" width="100%">
            <Box textAlign="center">
              <Image
                src="/imgs/feature1.png"
                alt="Echo Engine"
                boxSize="100px"
                mb="10px"
              />
              <Heading as="h3" size="md">
                Echo Engine
              </Heading>
              <Text fontSize="md">
                Classify animal vocalizations using cutting-edge machine
                learning models.
              </Text>
            </Box>
            <Box textAlign="center">
              <Image
                src="/imgs/feature2.png"
                alt="Echo API"
                boxSize="100px"
                mb="10px"
              />
              <Heading as="h3" size="md">
                Echo API
              </Heading>
              <Text fontSize="md">
                Seamlessly access and integrate data from various Echo
                components.
              </Text>
            </Box>
            <Box textAlign="center">
              <Image
                src="/imgs/feature3.png"
                alt="Echo Simulator"
                boxSize="100px"
                mb="10px"
              />
              <Heading as="h3" size="md">
                Echo Simulator
              </Heading>
              <Text fontSize="md">
                Simulate animal movements and sounds for real-time analysis.
              </Text>
            </Box>
          </Flex>
        </VStack>
      </Box>

      {/* Call to Action */}
      <Box
        as="section"
        width="100%"
        padding="50px 20px"
        backgroundColor="teal.500"
        color="white"
        textAlign="center"
      >
        <Heading as="h2" size="xl" mb="20px">
          Get Involved with Project Echo
        </Heading>
        <Text fontSize="lg" mb="30px">
          Join us in making a difference for wildlife conservation. Whether
          you're a developer, researcher, or conservationist, your contribution
          matters.
        </Text>
        <Button size="lg" colorScheme="whiteAlpha">
          Join Now
        </Button>
      </Box>
    </Flex>
  );
}
