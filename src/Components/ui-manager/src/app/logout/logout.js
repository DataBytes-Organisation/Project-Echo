"use client";

import React, { Component } from "react";

class ObjLogout extends Component {
  constructor(props) {
    super(props);
  }

  componentDidMount() {
    this.logoutUser();
  }

  async logoutUser() {
    try {
      const response = await fetch("/api/logout", {
        method: "POST",
        credentials: "include", // Ensures cookies are sent with the request
      });

      if (response.ok) {
        // Clear any stored token in localStorage or sessionStorage
        localStorage.removeItem("token");

        // Set a timeout to redirect to the home page after 5 seconds
        setTimeout(() => {
          window.location.href = "/login";
        }, 5000);
      } else {
        console.error("Failed to log out");
      }
    } catch (error) {
      console.error("An error occurred during logout:", error);
    }
  }

  render() {
    return (
      <div style={{ textAlign: "center", marginTop: "50px" }}>
        <p>Logging out...</p>
        <p>You will be redirected to the home page in 5 seconds...</p>
      </div>
    );
  }
}

export default ObjLogout;
