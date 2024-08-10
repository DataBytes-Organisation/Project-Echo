import { Box, Button, Text, Image, VStack } from "@chakra-ui/react";

import { isAuthenticated, getAuthStatus } from "../auth";
import ObjHome from "./ObjHome";

export default async function SoundPage() {
  const loggedInUser = await isAuthenticated();
  const userSession = await getAuthStatus();


  return (
    <VStack align="center" height="200px">
      {loggedInUser ? (
        <ObjHome />
      ) : (
        <p>Need to login</p>
      )}
    </VStack>
  );
}
