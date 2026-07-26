# Node.js and Nodemon Documentation

## Node.js

**Node.js** is an open-source, cross-platform, JavaScript runtime environment that executes JavaScript code outside of a web browser. It allows developers to use JavaScript to write command line tools and for server-side scripting—running scripts server-side to produce dynamic web page content before the page is sent to the user's web browser.

## Nodemon

**Nodemon** is a utility that will monitor for any changes in your source and automatically restart your server. It's perfect for development. Install it using `npm install -g nodemon`. Then you can just use `nodemon app.js` to run your application, and it will automatically restart when a file changes.

## Prestart in package.json

In `package.json`, you can specify scripts that can be run with the `npm run` command. The `prestart` script is a special script that runs automatically before the `start` script. This can be useful for setting up the environment or performing other necessary preconditions before starting your application.

## Running the project on a local sever

In the **terminal** opened in the project echo directory, use `cd \src\R and D\Project Echo Website\` to access the files of the project echo awareness website. Further, running the `npm run start` command in that terminal will install all the node.js dependencies automatically, while also hosting it onto **localhost:3000** local link. In this, any changes made into any files, except `server.js` will be reflected directly without **restarting the npm run start script**.