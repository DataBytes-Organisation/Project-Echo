"use client";

import { Box } from "@chakra-ui/react";

export default function MapPage() {

  return (
    <Box 
      width="100%" 
      height="calc(100vh - 180px)" 
      padding="20px" 
      boxShadow="lg" 
      borderRadius="md"
      backgroundColor="white"
    >
      <iframe
        src={"/mainmap"}
        title="Map"
        style={{ width: '100%', height: '100%', border: 'none', borderRadius: '8px' }}
      ></iframe>
    </Box>
  );
}
