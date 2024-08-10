import { Box, Button, Text, Image, VStack } from "@chakra-ui/react";
import LoginForm from "./login";
import { isAuthenticated, getAuthStatus } from "../auth";

export default async function LoginPage() {
  const loggedInUser = await isAuthenticated();
  const userSession = await getAuthStatus();

  console.log(userSession);

  return (
    <VStack align="center" height="200px">
      {loggedInUser ? (
        <Text color={"white"}>
          You are already logged in as {userSession[1]?.username}
        </Text>
      ) : (
        <LoginForm />
      )}
    </VStack>
  );
}
