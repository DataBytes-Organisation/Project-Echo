import React from "react";
import { useParams } from "react-router-dom";

const withParams = (Component) => {
  return (props) => {
    const params = useParams();
    return <Component {...props} params={params} />;
  };
};

export default withParams;
