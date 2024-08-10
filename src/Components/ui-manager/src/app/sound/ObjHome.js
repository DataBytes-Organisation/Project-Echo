"use client";


import React from "react";
import {
  Box,
  Button,
  Input,
  VStack,
  HStack,
  Heading,
  Text,
  Flex,
} from "@chakra-ui/react";
import { AddIcon } from "@chakra-ui/icons";
import { motion } from "framer-motion";


//import SimpleNAS from "./SimpleNAS";

class ObjHome extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      projects: [],
      newProjectName: "",
      accessRequests: [],
      currentProjectId: null,
    };
  }

  componentDidMount() {
    this.fetchProjects();
    //var t = new SimpleNAS();
  }

  fetchProjects = async () => {
    try {
      const response = await fetch("/api/projects", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
        },
      });
      const data = await response.json();
      this.setState({ projects: data.projects });
    } catch (error) {
      console.error("Error fetching projects:", error);
    }
  };

  handleAddProject = async () => {
    const { newProjectName } = this.state;
    if (newProjectName) {
      try {
        const response = await fetch("/api/projects", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
          },
          body: JSON.stringify({ name: newProjectName }),
        });
        const data = await response.json();
        this.setState((prevState) => ({
          projects: [...prevState.projects, data.project],
          newProjectName: "",
        }));
      } catch (error) {
        console.error("Error adding project:", error);
      }
    }
  };

  handleDeleteProject = async (id) => {
    // Add backend call to delete project if necessary
    this.setState((prevState) => ({
      projects: prevState.projects.filter((project) => project._id !== id),
    }));
  };

  handleInputChange = (e) => {
    this.setState({ newProjectName: e.target.value });
  };

  handleSelectProject = (project) => {
    if (project.isAccepted) {
      window.location.href = `/projects/${project._id}`;
    } else {
      console.log("You do not have access to this project.");
    }
  };

  handleRequestAccess = async (projectId) => {
    try {
      const response = await fetch(
        `/api/projects/request-access/${projectId}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
          },
        }
      );
      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      console.error("Error requesting access:", error);
    }
  };

  fetchAccessRequests = async (projectId) => {
    try {
      const response = await fetch(
        `/api/projects/access-requests/${projectId}`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
          },
        }
      );
      const data = await response.json();
      this.setState({
        accessRequests: data.accessRequests,
        currentProjectId: projectId,
      });
    } catch (error) {
      console.error("Error fetching access requests:", error);
    }
  };

  handleAcceptRequest = async (userId) => {
    const { currentProjectId } = this.state;
    try {
      const response = await fetch("/api/projects/accept-request", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
        },
        body: JSON.stringify({ projectId: currentProjectId, userId }),
      });
      const data = await response.json();
      console.log(data.message);
      this.fetchAccessRequests(currentProjectId); // Refresh access requests after accepting
    } catch (error) {
      console.error("Error accepting access request:", error);
    }
  };

  render() {
    const { projects, newProjectName, accessRequests } = this.state;

    return (
      <Flex align="center" justify="center" height="100vh" bg="gray.900">
        <Box
          width="350px"
          p="6"
          boxShadow="lg"
          borderRadius="md"
          bg="gray.800"
          color="white"
        >
          <VStack
            spacing={6}
            as={motion.div}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1 }}
          >
            <Heading as="h1" size="2xl" mb={4}>
              Projects
            </Heading>
            <HStack>
              <Input
                placeholder="New project name"
                value={newProjectName}
                onChange={this.handleInputChange}
                bg="white"
                color="black"
              />
              <Button
                onClick={this.handleAddProject}
                colorScheme="teal"
                leftIcon={<AddIcon />}
              >
                Add Project
              </Button>
            </HStack>
            {projects.length === 0 ? (
              <Text>No projects yet. Start by adding a new project.</Text>
            ) : (
              <VStack spacing={4} align="stretch" width="100%">
                {projects.map((project) => (
                  <Box
                    key={project._id}
                    p={2}
                    borderWidth="1px"
                    borderRadius="lg"
                    bg="gray.700"
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    cursor="pointer"
                    onClick={() => this.handleSelectProject(project)}
                  >
                    <Text>{project.name}</Text>
                    {!project.isAccepted && (
                      <Button
                        colorScheme="blue"
                        onClick={(e) => {
                          e.stopPropagation();
                          this.handleRequestAccess(project._id);
                        }}
                      >
                        Request Access
                      </Button>
                    )}
                    {project.isAccepted && (
                      <Button
                        colorScheme="purple"
                        onClick={(e) => {
                          e.stopPropagation();
                          this.fetchAccessRequests(project._id);
                        }}
                      >
                        View Requests
                      </Button>
                    )}
                  </Box>
                ))}
              </VStack>
            )}
          </VStack>
        </Box>
        {accessRequests.length > 0 && (
          <Box
            width="350px"
            p="6"
            boxShadow="lg"
            borderRadius="md"
            bg="gray.800"
            color="white"
            ml={4}
          >
            <VStack
              spacing={6}
              as={motion.div}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1 }}
            >
              <Heading as="h1" size="lg" mb={4}>
                Access Requests
              </Heading>
              {accessRequests.map((request) => (
                <Box
                  key={request._id}
                  p={2}
                  borderWidth="1px"
                  borderRadius="lg"
                  bg="gray.700"
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Text>{request.username}</Text>
                  <Button
                    colorScheme="teal"
                    onClick={() => this.handleAcceptRequest(request._id)}
                  >
                    Accept
                  </Button>
                </Box>
              ))}
            </VStack>
          </Box>
        )}
      </Flex>
    );
  }
}

export default ObjHome;
