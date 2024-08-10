import React from "react";
import { useNavigate } from "react-router-dom";

const withNavigation = (Component) => {
  return (props) => {
    const navigate = useNavigate();
    return <Component {...props} navigate={navigate} />;
  };
};

export default withNavigation;
