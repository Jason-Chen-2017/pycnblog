
作者：禅与计算机程序设计艺术                    

# 1.简介
  

The general goal of GraphQL mutations is to update the server-side state of a resource based on the input provided by the client. However, adding mutations to our schema requires additional work and knowledge. In this article, we will explore how to add mutations to our schema using Apollo Server and write an example application to demonstrate how it works in practice. We’ll start with some background information about mutations before diving into the specifics of implementing them.

# 2.Background
In traditional REST APIs, updates are typically implemented as HTTP PUT requests that modify existing resources or create new ones if they don't exist yet. In contrast, GraphQL mutations have several advantages:

1. They allow clients to specify precisely what changes should be made to the server-side data without having to worry about overloading GET requests or creating separate endpoints for each action.
2. The syntax used by GraphQL mutations closely matches that of queries, making it easier to learn and remember.
3. By separating read operations from mutation operations, GraphQL can provide more efficient performance because any side effects (such as writing to disk) won't occur until explicitly requested by the client.

In addition to updating server-side data, GraphQL mutations also support other actions such as deleting records, executing complex business logic like payments or email notifications, and more. 

To implement GraphQL mutations in our schema, we need to define types that represent the different mutations we want to expose and then map those types to resolver functions that actually execute the desired behavior when the corresponding mutation operation is executed. These resolver functions take the input arguments specified by the client and perform the necessary modifications to the server-side state of the underlying database or other data store.

Let's dive deeper into the specifics of defining mutations, mapping them to resolver functions, and handling authentication requirements in order to complete the implementation of an example application.

# 3.Basic Concepts & Terminology
Before we move forward, let's quickly review some basic concepts related to GraphQL mutations and terminology that you may encounter while reading through this article. 

1. **Mutation**: A mutation in GraphQL refers to a type of operation that creates, updates, or deletes a resource on the server. It is similar to a query but instead of returning data, it modifies server-side state. Mutation types are defined within the "mutation" section of the schema language.

2. **Resolver Functions**: Resolver functions are responsible for resolving the actual execution of the mutation request against the server-side data storage layer. Each resolver function receives the input parameters passed by the client and performs the required modification(s).

3. **Input Types**: Input types are custom scalar types used by fields in our schema that accept arguments. These types define the structure of the expected input values and enable us to validate user inputs before proceeding with the mutation. For instance, we might define an input type called "AddProductInput" with properties like name, price, description, etc., which could be used as an argument for a field that adds a product to our inventory system.

4. **Context Argument**: Context argument provides access to global metadata about the current request, including headers, cookies, and variables. This allows us to implement features like authorization checks or logging at the resolver level.

5. **Authentication Requirements**: Authentication requirements usually involve verifying that a user has the correct permissions to make a particular mutation request. There are many ways to handle authentication requirements in GraphQL, depending on your use case and preferred security model. Some popular approaches include JWT tokens, OAuth 2.0, and API keys.

Now that we've reviewed the basics, let's get started by setting up our development environment. We'll be using Node.js with the following dependencies:

- apollo-server - An open-source library created by Apollo that enables us to easily set up a GraphQL server with TypeScript integration and comes pre-configured with tools like loading data sources, integrating middleware, and caching responses.
- graphql - A powerful and flexible JavaScript library for buildingGraphQL schemas and integrates well with Express/Koa frameworks.
- mongoose - MongoDB object modeling tool designed to work in an asynchronous environment.
- dotenv - A zero-dependency module that loads environment variables from.env files into process.env.

We'll also be using MongoDB Atlas to host our database, which provides a free tier option that includes limited usage of their cloud infrastructure. Once we're all set up, we can begin implementing our example application step by step.

# Step 1: Setting Up the Environment

First, let's install the required dependencies and set up the project folder. Open a terminal window and run the following commands:

```bash
mkdir adding-mutations && cd adding-mutations
npm init -y # initialize package.json file
npm i apollo-server graphql mongoose dotenv
touch index.ts
```

This will create a new directory called `adding-mutations`, switch to its root folder, and install the required packages. We'll also create three empty files: `index.ts` where we'll place our code, `.gitignore` to keep track of untracked files, and `README.md` to document our project. Finally, we'll open `index.ts` in our editor of choice to get started.


Next, we need to set up our environment variables. Create a new file named `.env` and add the following lines:

```bash
DB_CONNECTION=mongodb+srv://<username>:<password>@<cluster>/<dbname>?retryWrites=true&w=majority
PORT=<your port>
```

Replace `<username>`, `<password>`, `<cluster>`, and `<dbname>` with your own Mongo DB Atlas connection details. You can find these details in the Mongo DB Atlas dashboard after clicking on "Connect". Your cluster URL will look something like this: `mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>?retryWrites=true&w=majority`. Replace `<your port>` with the port number you wish to run the server on. 

Note: If you're testing locally, replace `mongodb+srv://...` with `mongodb://localhost:<port>/`.

# Step 2: Creating a Mongoose Data Source

Our first task is to connect our server to a MongoDB database. To do this, we'll need to create a data source using Mongoose, a popular ODM (Object Document Mapper) for working with MongoDB databases in Node.js. Let's create a new file called `db.ts` in our `/src` directory and add the following code:

```typescript
import * as mongoose from'mongoose';
import { resolve } from 'path';
const env = require('dotenv').config({ path: resolve(__dirname, '../', '.env') }).parsed;

export async function connect() {
  const db = await mongoose.connect(`mongodb://${env.DB_CONNECTION}`, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  console.log(`Connected to ${db.connections[0].name} @ ${db.connections[0].host}`);
  return db;
}

export default mongoose;
```

Here, we import `mongoose` and load our environment variables using `dotenv` module. Then, we export two functions - `connect()` and `default`, where `connect()` returns a promise that resolves to the connected `mongoose` instance and `default` exports the `mongoose` module itself.

Inside `connect()`, we construct a MongoDB connection string using the value of `DB_CONNECTION` variable, which contains the username, password, cluster endpoint, and database name. We pass the constructed connection string to the `mongoose.connect()` method along with options for MongoDB driver configuration. Finally, we log out the status message indicating the successfull connection. Note that we wrap this functionality inside an asynchronous function so that we can use `await` keyword to wait for the connection to establish before moving on to the next step.

Once we have successfully established a connection to our database, we can continue with defining our schema.

# Step 3: Defining a Product Type

A common pattern in GraphQL is to define types that correspond to the various objects that we want to expose via our API. Here, we'll define a simple `Product` type that represents a single item in our catalogue. Create a new file called `product.ts` in our `/src` directory and add the following code:

```typescript
import { gql } from 'apollo-server';

const typeDefs = gql`
  type Product {
    id: ID!
    name: String!
    description: String
    price: Float!
  }
`;

export default typeDefs;
```

This defines a GraphQL type called `Product` with four properties: `id` (a unique identifier), `name` (the title of the product), `description` (a brief summary of the product), and `price` (its selling price). Note that we prepend the `gql` template tag to indicate that this is a GraphQL schema definition written in the GraphQL Schema Definition Language (SDL).

We can now import this `typeDefs` constant and register it with our GraphQL server.

# Step 4: Implementing CRUD Operations

One of the most important tasks in any web application is to enable users to create, read, update, and delete data. With GraphQL, implementing these operations is easy since we just define the appropriate mutation types and associate them with resolver functions that interact with our MongoDB data source. Here, we'll implement the "create", "read", "update", and "delete" operations on the `Product` type.

Create a new file called `resolvers.ts` in our `/src` directory and add the following code:

```typescript
import Product from './models/product';
import { IResolvers } from 'graphql-tools';

const resolvers: IResolvers = {
  Query: {},
  Mutation: {
    async createProduct(_, args): Promise<any> {
      try {
        const product = await Product.create(args);

        return {
          ok: true,
          product,
        };
      } catch (err) {
        return err;
      }
    },
    async updateProduct(_, args): Promise<any> {
      try {
        const updatedProduct = await Product.findByIdAndUpdate(args._id, args.data, { new: true });

        return {
          ok: true,
          product: updatedProduct,
        };
      } catch (err) {
        return err;
      }
    },
    async deleteProduct(_, args): Promise<any> {
      try {
        const deletedProduct = await Product.findByIdAndDelete(args._id);

        return {
          ok: true,
          product: deletedProduct,
        };
      } catch (err) {
        return err;
      }
    },
  },
};

export default resolvers;
```

This defines four resolver functions (`createProduct()`, `updateProduct()`, `deleteProduct()`) that respond to the `createProduct`, `updateProduct`, and `deleteProduct` mutation types defined earlier. The `async`/`await` keywords allow us to write cleaner code using promises rather than callbacks.

Each resolver function takes two arguments - `_` and `args`. `_` corresponds to the parent context object, which is not needed here. Instead, `args` is an object containing the arguments passed by the client in the form `{ _id: "<mongoId>", data: {...} }` where `_id` is the MongoDB ObjectID of the record to be modified and `data` is an object containing the key-value pairs representing the fields to be updated. We extract these arguments using destructuring assignment and pass them directly to the respective Mongoose methods (`create()`, `findByIdAndUpdate()`, and `findByIdAndDelete()`) to perform the appropriate data manipulations.

If the method call succeeds, we return an object containing a boolean property `ok` set to `true` and the modified or deleted product document. Otherwise, we return the error thrown by the MongoDB driver.

Finally, we export the `resolvers` array containing the resolver functions associated with our mutation types.

Note that we haven't defined the `Product` class yet, so let's create another file called `models/product.ts` in our `/src` directory and add the following code:

```typescript
import mongoose, { Schema } from'mongoose';

const productSchema = new Schema({
  name: { type: String, required: [true, 'Name is required'] },
  description: { type: String },
  price: { type: Number, required: [true, 'Price is required'] },
});

const ProductModel = mongoose.model('Product', productSchema);

class Product extends ProductModel {}

module.exports = Product;
```

This defines a Mongoose schema for our `Product` type and registers it with the MongoDB collection named `"Product"` using the `model()` method. We also define a subclass `Product` that inherits from the generated Mongoose model and exports it.

With these steps completed, we're ready to test our GraphQL mutations end-to-end.

# Step 5: Testing the Application

Let's put everything together and see how our application responds to GraphQL mutation requests. Create a new file called `app.ts` in our `/src` directory and add the following code:

```typescript
import express from 'express';
import { ApolloServer } from 'apollo-server-express';
import bodyParser from 'body-parser';
import resolvers from './resolvers';
import typeDefs from './product';
import db from './db';

const app = express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: ({ req }) => ({
    // Add auth validation or other middlewares here
  }),
  introspection: true,
  playground: true,
});

// Start up DB connection
db.connect().then(() => {
  server.applyMiddleware({ app });
  app.listen(process.env.PORT || 4000, () => {
    console.log(`Server running on http://localhost:${process.env.PORT || 4000}/`);
  });
}).catch((error) => console.error(error));
```

Here, we instantiate an `Express` server and apply the `ApolloServer` middleware, passing in our schema definitions and resolver functions. We also configure middleware for parsing JSON and URL-encoded payloads. Next, we start up our database connection by calling the `connect()` function exported from our `db.ts` file. Finally, we listen to the configured port (defaults to `4000`) and output a status message indicating whether the server was successfully initialized.

At this point, we can start up the server by running the following command:

```bash
nodemon index.ts
```

When the server starts, we should see messages indicating that the server is listening on localhost and the port we configured. Now, we can send GraphQL mutation requests to the server to create, read, update, and delete products. For example, the following CURL command would create a new product with the given name and price:

```bash
curl --request POST \
     --url http://localhost:4000/ \
     --header 'Content-Type: application/json' \
     --data '{
         "query": "mutation {
           createProduct(input: {
             name: \"iPhone X\", 
             price: 999.99
           }) {
             ok
             product {
               id
               name
               price
             }
           } 
         }"
       }'
```

This would return a response containing a `product` object with the newly created product details. Similarly, the following CURL command would retrieve the list of all products stored in the database:

```bash
curl --request POST \
     --url http://localhost:4000/ \
     --header 'Content-Type: application/json' \
     --data '{
         "query": "{
           products {
             id
             name
             price
           } 
         }"
       }'
```

Similarly, this command would return a response containing an array of `products` objects, one for each product in our catalogue. Lastly, the following command demonstrates how to update an existing product:

```bash
curl --request POST \
     --url http://localhost:4000/ \
     --header 'Content-Type: application/json' \
     --data '{
         "query": "mutation {
           updateProduct(input: {
             _id: \"<mongoId>\",
             data: {
               name: \"Samsung Galaxy S7\",
               price: 599.99
             }
           }) {
             ok
             product {
               id
               name
               price
             }
           } 
         }"
       }'
```

Again, this command would return a response containing a `product` object with the updated product details. Finally, the following command shows how to delete a product:

```bash
curl --request POST \
     --url http://localhost:4000/ \
     --header 'Content-Type: application/json' \
     --data '{
         "query": "mutation {
           deleteProduct(input: {
             _id: \"<mongoId>\"
           }) {
             ok
             product {
               id
               name
               price
             }
           } 
         }"
       }'
```

This command would return a response containing a `product` object with the deleted product details.