"use client";

import React, { Component } from "react";
import { Box, Button, Input, VStack, Text, Flex } from "@chakra-ui/react";
//import { Navigate } from "react-router-dom";

class ObjLogin extends Component {
  constructor(props) {
    super(props);
    this.state = {
      username: "",
      password: "",
      message: "",
      loggedInUser: null,
      redirectToHome: false,
    };

    this.handleInputChange = this.handleInputChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  async componentDidMount() {
    const token = localStorage.getItem("token");
    if (token) {
      const response = await fetch("/api/check-auth", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        this.setState({ loggedInUser: data.username });
      }
    }
  }

  handleInputChange(event) {
    const { name, value } = event.target;
    this.setState({
      [name]: value,
    });
  }

  async handleSubmit(event) {
    event.preventDefault();

    const { username, password } = this.state;

    const response = await fetch("/api/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();

    if (response.ok) {
      this.setState({
        message: "Login successful",
        loggedInUser: data.username,
        redirectToHome: true,
      });
      localStorage.setItem("token", data.token);
    } else {
      this.setState({ message: `Error: ${data.error}` });
    }
  }

  render() {
    const { username, password, message, loggedInUser, redirectToHome } =
      this.state;

    if (redirectToHome) {
      window.location.href = "/";
    }

    return (
      <Flex align="center" justify="center" height="100vh">
        <Box
          width="350px"
          p="6"
          boxShadow="lg"
          borderRadius="md"
          bg="gray.800"
          color="white"
        >
          {loggedInUser && (
            <Text fontSize="xl" mb="4" textAlign="center">
              Logged in as: {loggedInUser}
            </Text>
          )}
          <Text fontSize="2xl" mb="6" textAlign="center">
            Login
          </Text>
          <form onSubmit={this.handleSubmit}>
            <VStack spacing="4">
              <Input
                placeholder="Username"
                name="username"
                value={username}
                onChange={this.handleInputChange}
                bg="gray.700"
                border="1px solid"
                borderColor="gray.600"
                _placeholder={{ color: "gray.400" }}
              />
              <Input
                placeholder="Password"
                type="password"
                name="password"
                value={password}
                onChange={this.handleInputChange}
                bg="gray.700"
                border="1px solid"
                borderColor="gray.600"
                _placeholder={{ color: "gray.400" }}
              />
              <Button
                type="submit"
                colorScheme="blue"
                width="full"
                _hover={{ bg: "blue.600" }}
              >
                Login
              </Button>
            </VStack>
          </form>
          {message && (
            <Text mt="4" textAlign="center" color="red.400">
              {message}
            </Text>
          )}
        </Box>
      </Flex>
    );
  }
}

export default ObjLogin;
