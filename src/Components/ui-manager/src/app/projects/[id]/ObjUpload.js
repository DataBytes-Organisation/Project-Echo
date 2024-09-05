"use client"


import React, { Component } from 'react';

class ObjUpload extends Component {
  constructor(props) {
    super(props);
    this.state = {
      fileCount: 0,
      totalSizeMB: 0,
      files: [],
    };
  }

  handleFolderSelect = async (event) => {
    const files = event.target.files;
    let filesArray = [];
    let totalSize = 0;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      filesArray.push(file);
      totalSize += file.size; // Accumulate file size in bytes
      console.log('File:', file.name); // Log each file
    }

    const totalFiles = filesArray.length;
    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2); // Convert bytes to MB and round to 2 decimal places

    this.setState({ fileCount: totalFiles, totalSizeMB: totalSizeMB, files: filesArray });

    console.log('Total files to be uploaded:', totalFiles);
    console.log('Total size (MB):', totalSizeMB);
  };

  render() {
    return (
      <div>
        <input
          type="file"
          webkitdirectory="true"
          directory="true"
          onChange={this.handleFolderSelect}
          style={{ display: 'block', marginBottom: '20px' }}
        />
        <div>Total Files to be uploaded: {this.state.fileCount}</div>
        <div>Total Size: {this.state.totalSizeMB} MB</div>
      </div>
    );
  }
}

export default ObjUpload;
