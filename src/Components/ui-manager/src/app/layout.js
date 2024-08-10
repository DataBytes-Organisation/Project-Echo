import { Inter } from "next/font/google";
import "./globals.css";
import {
  ChakraProvider,
  Box,
  Button,
  Text,
  Image,
  Flex,
  Spacer,
} from "@chakra-ui/react";
import Link from "next/link";
import { cookies } from "next/headers";
import { isAuthenticated } from "./auth";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Project Echo",
  description: "Project Echo",
};

export default async function RootLayout({ children }) {
  const cookieStore = cookies();
  const token = cookieStore.get("token")?.value || null;
  const loggedInUser = await isAuthenticated(token);

  return (
    <html lang="en">
      <body className={inter.className}>
        <ChakraProvider>
          <Flex
            direction="column"
            minHeight="100vh"
            position="relative"
            _before={{
              content: '""',
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundImage: `url(/imgs/bg.jpg)`,
              backgroundSize: "cover",
              backgroundRepeat: "no-repeat",
              backgroundPosition: "center",
              filter: "blur(8px)", /* Blur effect */
              zIndex: "-1",
            }}
          >
            <Box flex="1" position="relative" zIndex="1">
              <Box
                as="nav"
                padding="10px"
                backgroundColor="rgba(0, 0, 0, 0.7)"
                color="white"
                w="100%"
                p="20px"
                boxShadow="0 4px 6px rgba(0, 0, 0, 0.1)"
                display="flex"
                alignItems="center"
              >
                <Flex align="center">
                  <Link href="/">
                    <Button colorScheme="teal" variant="outline" mr="4">
                      Home
                    </Button>
                  </Link>

                  <Link href="/map">
                    <Button colorScheme="teal" variant="outline" mr="4">
                      Map
                    </Button>
                  </Link>
                  <Link href="/sound">
                    <Button colorScheme="teal" variant="outline" mr="4">
                      Train sound
                    </Button>
                  </Link>
                </Flex>

                <Spacer />

                <Flex align="center">
                  {loggedInUser ? (
                    <>
                      <Link href="/profile" passHref>
                        <Button
                          as="span"
                          colorScheme="teal"
                          variant="outline"
                          mr="4"
                        >
                          Profile
                        </Button>
                      </Link>
                      <Link href="/logout" passHref>
                        <Button as="span" colorScheme="teal" variant="outline">
                          Logout
                        </Button>
                      </Link>
                    </>
                  ) : (
                    <>
                      <Link href="/login" passHref>
                        <Button
                          as="span"
                          colorScheme="teal"
                          variant="outline"
                          mr="4"
                        >
                          Login
                        </Button>
                      </Link>
                      <Link href="/register" passHref>
                        <Button
                          as="span"
                          colorScheme="teal"
                          variant="outline"
                          mr="4"
                        >
                          Register
                        </Button>
                      </Link>
                    </>
                  )}
                </Flex>
              </Box>
              <Box position="relative" zIndex="1" paddingBottom="100px">
                {children}
              </Box>
            </Box>
            <Box
              as="footer"
              width="100%"
              backgroundColor="rgba(0, 0, 0, 0.7)"
              color="white"
              padding="10px"
              textAlign="center"
            >
              <Text>
                © {new Date().getFullYear()} Project Echo. All rights reserved.
              </Text>
            </Box>
          </Flex>
        </ChakraProvider>
      </body>
    </html>
  );
}
