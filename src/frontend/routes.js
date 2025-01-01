import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import SightingUploadForm from '../Components/sighting_upload_form.js';

const Routes = () => (
  <Router>
    <Switch>
      <Route exact path="/" component={HomePage} />
      <Route path="/upload-sighting" component={SightingUploadForm} />
    </Switch>
  </Router>
);

export default Routes;