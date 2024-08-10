"use client";


import React, { Component } from "react";
import {
  Box,
  Input,
  VStack,
  Text,
  Button,
  HStack,
  Select,
  List,
  ListItem,
} from "@chakra-ui/react";

class ObjLabelManager extends Component {
  constructor(props) {
    super(props);
    this.state = {
      category: "",
      subcategory: "",
      subsubcategory: "",
      selectedCategory: "",
      selectedSubcategory: "",
      data: {},
    };
  }

  componentDidMount() {
    if (this.props.projectId) {
      this.fetchLabels();
    }
  }

  componentDidUpdate(prevProps) {
    if (this.props.projectId !== prevProps.projectId && this.props.projectId) {
      this.fetchLabels();
    }
  }

  fetchLabels = async () => {
    try {
      const response = await fetch(
        `/api/projects/${this.props.projectId}/labels`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
          },
        }
      );
      const data = await response.json();
      if (data.labels) {
        this.props.sendDataToParent(data.labels);
        this.setState({ data: data.labels });
      }
    } catch (error) {
      console.error("Error fetching labels:", error);
    }
  };

  saveLabels = async (labels) => {
    try {
      const response = await fetch(
        `/api/projects/${this.props.projectId}/labels`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${document.cookie.replace("token=", "")}`,
          },
          body: JSON.stringify({ labels }),
        }
      );
      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      console.error("Error saving labels:", error);
    }
  };

  handleCategoryChange = (event) => {
    this.setState({ category: event.target.value });
  };

  handleSubcategoryChange = (event) => {
    this.setState({ subcategory: event.target.value });
  };

  handleSubsubcategoryChange = (event) => {
    this.setState({ subsubcategory: event.target.value });
  };

  handleSelectedCategoryChange = (event) => {
    this.setState({
      selectedCategory: event.target.value,
      selectedSubcategory: "",
    });
  };

  handleSelectedSubcategoryChange = (event) => {
    this.setState({ selectedSubcategory: event.target.value });
  };

  addLabel = () => {
    const { category, subcategory, subsubcategory, data } = this.state;

    if (!category) return;

    const newData = { ...data };

    if (!newData[category]) {
      newData[category] = {};
    }

    if (subcategory) {
      if (!newData[category][subcategory]) {
        newData[category][subcategory] = [];
      }

      if (subsubcategory) {
        newData[category][subcategory].push(subsubcategory);
      }
    }
    this.props.sendDataToParent(newData);
    this.saveLabels(newData);
    this.setState({
      data: newData,
      category: "",
      subcategory: "",
      subsubcategory: "",
    });
  };

  renderNestedData = (data) => {
    return (
      <List spacing={3}>
        {Object.keys(data).map((category) => (
          <ListItem key={category}>
            <Text fontWeight="bold">{category}</Text>
            {typeof data[category] === "object" &&
              this.renderNestedData(data[category])}
            {Array.isArray(data[category]) && (
              <List spacing={1} pl={4}>
                {data[category].map((item) => (
                  <ListItem key={item}>{item}</ListItem>
                ))}
              </List>
            )}
          </ListItem>
        ))}
      </List>
    );
  };

  render() {
    const {
      category,
      subcategory,
      subsubcategory,
      selectedCategory,
      selectedSubcategory,
      data,
    } = this.state;

    const subcategories = selectedCategory
      ? Object.keys(data[selectedCategory] || {})
      : [];
    const subsubcategories =
      selectedCategory && selectedSubcategory
        ? data[selectedCategory][selectedSubcategory] || []
        : [];

    return (
      <Box>
        <VStack spacing={4} align="stretch">
          <HStack>
            <Input
              placeholder="Category"
              value={category}
              onChange={this.handleCategoryChange}
            />
            <Input
              placeholder="Subcategory"
              value={subcategory}
              onChange={this.handleSubcategoryChange}
            />
            <Input
              placeholder="Sub-subcategory"
              value={subsubcategory}
              onChange={this.handleSubsubcategoryChange}
            />
            <Button onClick={this.addLabel}>Add</Button>
          </HStack>
          <Box mt={4}>
            <HStack>
              <Select
                placeholder="Select Category"
                value={selectedCategory}
                onChange={this.handleSelectedCategoryChange}
                bg="teal"
                borderColor="teal"
                color="white"
              >
                {Object.keys(data).map((category) => (
                  <option
                    key={category}
                    value={category}
                    style={{ backgroundColor: "teal" }}
                  >
                    {category}
                  </option>
                ))}
              </Select>
              <Select
                placeholder="Select Subcategory"
                value={selectedSubcategory}
                onChange={this.handleSelectedSubcategoryChange}
                isDisabled={!selectedCategory}
                bg="teal"
                borderColor="teal"
                color="white"
              >
                {subcategories.map((subcategory) => (
                  <option
                    key={subcategory}
                    value={subcategory}
                    style={{ backgroundColor: "lightgreen" }}
                  >
                    {subcategory}
                  </option>
                ))}
              </Select>
              <Select
                placeholder="Select Sub-subcategory"
                isDisabled={!selectedSubcategory}
                bg="teal"
                borderColor="teal"
                color="white"
              >
                {subsubcategories.map((subsubcategory) => (
                  <option
                    key={subsubcategory}
                    value={subsubcategory}
                    style={{ backgroundColor: "lightcoral" }}
                  >
                    {subsubcategory}
                  </option>
                ))}
              </Select>
            </HStack>
          </Box>
          <Box>
            {Object.keys(data).length > 0 && (
              <Box mt={4}>
                <Text mb={2}>Labels:</Text>
                {this.renderNestedData(data)}
              </Box>
            )}
          </Box>
        </VStack>
      </Box>
    );
  }
}

export default ObjLabelManager;
