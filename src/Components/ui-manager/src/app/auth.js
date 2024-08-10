import http from "http";
import fetch from "node-fetch";
import { cookies } from "next/headers";

export async function checkAuth(token) {
  if (!token) return null;

  // Use environment variable to determine the base URL
  const baseUrl = process.env.API_BASE_URL || "http://localhost:3123"; // Defaults to localhost if not set

  const url = `${baseUrl}/api/check-auth`;

  const agent = new http.Agent({ rejectUnauthorized: false });

  try {
    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
      agent,
    });
    if (response.ok) {
      const data = await response.json();
      return data.username;
    }
  } catch (error) {
    console.error("Error checking authentication:", error);
  }
  return null;
}

export function getTokenFromCookies() {
  const cookieStore = cookies();
  const token = cookieStore.get("token");
  return token ? token.value : null;
}

export async function isAuthenticated() {
  const token = getTokenFromCookies();
  const loggedInUser = token ? await checkAuth(token) : null;
  return !!loggedInUser;
}

export async function getAuthStatus() {
  const token = getTokenFromCookies();
  const loggedInUser = token ? await checkAuth(token) : null;

  if (loggedInUser) {
    const baseUrl = process.env.API_BASE_URL || "http://localhost:3123"; // Defaults to localhost if not set

    const url = `${baseUrl}/api/check-auth`;
    const agent = new http.Agent({ rejectUnauthorized: false });

    try {
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        agent,
      });
      if (response.ok) {
        const userData = await response.json();
        return [true, userData];
      }
    } catch (error) {
      console.error("Error fetching user data:", error);
    }
  }

  return [false, {}];
}
