"use client";


import React, { Component } from "react";
import { Box, VStack, Select, Text, Tooltip, Flex } from "@chakra-ui/react";

class ObjSelectLabel extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedCategory: "",
      selectedSubcategory: "",
      selectedSubsubcategory: "",
    };
  }

  handleCategoryChange = (event) => {
    this.props.sendDataToParent({
      uniqPassIn: this.props.uniqPassIn,
      uniqPassFile: this.props.uniqPassFile,
      lstCats: [
        event.target.value,
        this.state.selectedSubcategory,
        this.state.selectedSubsubcategory,
      ],
    });

    this.setState({
      selectedCategory: event.target.value,
      selectedSubcategory: "",
      selectedSubsubcategory: "",
    });
  };

  handleSubcategoryChange = (event) => {
    this.props.sendDataToParent({
      uniqPassIn: this.props.uniqPassIn,
      uniqPassFile: this.props.uniqPassFile,
      lstCats: [
        this.state.selectedCategory,
        event.target.value,
        this.state.selectedSubsubcategory,
      ],
    });

    this.setState({
      selectedSubcategory: event.target.value,
      selectedSubsubcategory: "",
    });
  };

  handleSubsubcategoryChange = (event) => {
    this.props.sendDataToParent({
      uniqPassIn: this.props.uniqPassIn,
      uniqPassFile: this.props.uniqPassFile,
      lstCats: [
        this.state.selectedCategory,
        this.state.selectedSubcategory,
        event.target.value,
      ],
    });

    this.setState({ selectedSubsubcategory: event.target.value });
  };

  render() {
    const { data } = this.props;
    const { selectedCategory, selectedSubcategory, selectedSubsubcategory } =
      this.state;

    const categories = Object.keys(data);
    const subcategories = selectedCategory
      ? Object.keys(data[selectedCategory] || {})
      : [];
    const subsubcategories =
      selectedCategory && selectedSubcategory
        ? data[selectedCategory][selectedSubcategory] || []
        : [];

    return (
      <Box>
        <Flex>
          <Select
            placeholder="C1"
            value={selectedCategory}
            onChange={this.handleCategoryChange}
            bg="teal"
            borderColor="teal"
            color="white"
          >
            {categories.map((category) => (
              <Tooltip
                label={category}
                aria-label="Category Tooltip"
                key={category}
              >
                <option
                  key={category}
                  value={category}
                  style={{ backgroundColor: "teal" }}
                >
                  {category.length > 10
                    ? `${category.substring(0, 10)}...`
                    : category}
                </option>
              </Tooltip>
            ))}
          </Select>
          <Select
            placeholder="C2"
            value={selectedSubcategory}
            onChange={this.handleSubcategoryChange}
            isDisabled={!selectedCategory}
            bg="teal"
            borderColor="teal"
            color="white"
          >
            {subcategories.map((subcategory) => (
              <Tooltip
                label={subcategory}
                aria-label="Subcategory Tooltip"
                key={subcategory}
              >
                <option
                  key={subcategory}
                  value={subcategory}
                  style={{ backgroundColor: "lightgreen" }}
                >
                  {subcategory.length > 10
                    ? `${subcategory.substring(0, 10)}...`
                    : subcategory}
                </option>
              </Tooltip>
            ))}
          </Select>
          <Select
            placeholder="C3"
            value={selectedSubsubcategory}
            onChange={this.handleSubsubcategoryChange}
            isDisabled={!selectedSubcategory}
            bg="teal"
            borderColor="teal"
            color="white"
          >
            {subsubcategories.map((subsubcategory) => (
              <Tooltip
                label={subsubcategory}
                aria-label="Sub-subcategory Tooltip"
                key={subsubcategory}
              >
                <option
                  key={subsubcategory}
                  value={subsubcategory}
                  style={{ backgroundColor: "lightcoral" }}
                >
                  {subsubcategory.length > 10
                    ? `${subsubcategory.substring(0, 10)}...`
                    : subsubcategory}
                </option>
              </Tooltip>
            ))}
          </Select>
        </Flex>
      </Box>
    );
  }
}

export default ObjSelectLabel;
