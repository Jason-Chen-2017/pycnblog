
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Apache Impala (incubating) is a popular open-source distributed SQL query engine that provides high performance for large datasets stored in HDFS or Amazon S3 file systems. It has been widely used in enterprise data warehouses to process large volumes of data with low latency and high throughput. However, securing access to the database server is becoming increasingly critical as organizations adopt cloud computing. In this article, we will discuss how to secure an Apache Impala deployment on both standalone and clustered mode using authentication and authorization features available in Impala. The focus is not only on enabling user authentication but also ensuring that users have access to resources they are authorized to access within Impala.
         
         This article assumes some familiarity with basic concepts such as Hadoop, Linux, networking, databases, and web security. If you're new to these areas, please review relevant materials before proceeding further.
         
         # 2.核心概念
         ## 2.1 Kerberos Authentication Protocol
         
         Kerberos is an industry-standard protocol developed by the National Institute of Standards and Technology (NIST). It's designed to provide strong authentication between computer systems across an organization's network boundary without requiring direct shared secrets.
         
         The key idea behind Kerberos is to use tickets instead of passwords when authenticating users to services. A ticket represents a limited-use session key that can be used to authenticate a client to a server after verifying its identity through mutual authentication. Tickets are short-lived, which means they expire quickly after being issued to ensure that they cannot be replayed. To obtain a valid ticket, users need to present their credentials, typically a username and password, to a KDC (Key Distribution Center), which validates them and issues a ticket. The ticket then needs to be presented back to the KDC to request access to the protected resource. When accessing a service, clients present the appropriate ticket to establish authenticated sessions with servers.
         
         Here's an illustration of how Kerberos works:
         
         
         
         
         
         Fig. 1 - Illustration of Kerberos Architecture
         
         
         ## 2.2 User Authentication and Authorization Models
         
         There are two main models for user authentication and authorization in Impala:
         
         1. Allow all access
         
            By default, any user who connects to the Impala daemon can run queries without having to authenticate themselves first. In simple deployments, this may be sufficient since it makes it easier to get started with Impala. However, it's important to note that allowing all access comes at the risk of exposing sensitive information to unauthorized users.
         
            
         2. LDAP/Kerberos based authentication and authorization
         
            With Impala clusters running in production, we recommend using either LDAP or Kerberos authentication to control access to the Impala server and allow only authorized users to execute queries against the system.
            
            LDAP (Lightweight Directory Access Protocol) is a directory service protocol that enables centralized management of user identities. It allows administrators to manage user accounts, group membership, and permissions centrally, making it easy to maintain consistent policy across multiple applications and services. For example, Active Directory (AD) is an LDAP-based implementation that Microsoft uses to manage domain controllers.
        
            Kerberos authentication involves exchanging a set of encrypted messages to prove your identity to a server, while authorization refers to determining what privileges you have to access specific resources on the server. Administrators configure policies in the KDC to define what users are allowed to connect to the server and what actions each user is allowed to perform. Each time a user attempts to access a resource on the server, they must present their credential (typically a username and password combination) to the KDC for validation. The KDC checks the validity of the credentials and returns a ticket if they are correct. Once the user presents the ticket to the server, the server decrypts it and verifies that it is valid and grants access according to the configured policies.
        
            Overall, LDAP/Kerberos-based authentication and authorization offer several benefits over the "allow all" approach:
            
            1. Role-based access control (RBAC): This model allows administrators to specify granular access controls based on roles assigned to users, groups, and other entities within the organization. For example, an analyst might have read-only access to certain tables while a developer might have full write access.
             
            2. Centralized administration: Instead of granting access rights individually to individual users or machines, LDAP/Kerberos-based authentication and authorization relies on a centralized configuration that can be managed easily from one location.
              
             
            Additionally, Impala supports external authentication mechanisms such as OAuth2, which enable third-party authentication providers like Google or Facebook to integrate with Impala. These mechanisms allow for more flexibility in managing access across multiple platforms and devices.
         
         
       
          
      
         
     

         ## 2.3 Roles and Privileges
         
         In addition to authentication and authorization, Impala defines several built-in roles and associated privileges that determine what operations users can perform within the system. The following table summarizes the different types of roles and their associated privileges:
         
         
         
         
         Table 1 - Impala Built-in Roles and Associated Privileges
         
         | Role Name | Description | Associated Privileges |
         |:-------:|:-----------:|:----------------------:|
         | All | Grants full read and write access to all objects in the catalog including schemas, tables, functions, views, etc. | CREATE, DROP, SELECT, INSERT, UPDATE, DELETE, SHOW_DATABASES, ALL PRIVILEGES ON ALL TABLES IN SCHEMA, REFRESH METADATA |
         | Admin | Allows users to modify metadata such as creating, dropping, altering tables and views, adding or removing columns, creating functions, views, etc., and setting configurations. It also allows access to various system tables such as the EVENTS table containing audit logs of changes made to the system. | ALTER, DROP, CREATE, INSERT, DELETE, SELECT, UPDATE, REFERENCES, TRIGGER, ALL PRIVILEGES ON ALL TABLES IN SCHEMA, SYSTEM ADMINISTRATION |
         | Insert | Allows users to insert data into tables. | INSERT |
         | Refresh | Allows users to refresh the cache for tables and views. | REFRESH |
         | Select | Allows users to select data from tables and views. | SELECT |
         | Update | Allows users to update existing rows in tables and views. | UPDATE |
         | Delete | Allows users to delete rows from tables and views. | DELETE |
         
         
         Note that Impala does not currently support role inheritance, so users with privileges granted via higher-level roles do not automatically inherit those privileges. If a user wants to take advantage of inherited privileges, they should explicitly assign them to themselves or create custom roles with additional privileges.
         
         
         ### 2.4 Databases, Tables, and Views
         
         As mentioned earlier, Impala stores metadata about databases, tables, and views in a centralized catalog called the Catalog Service. Every object created in Impala belongs to a specific database, which serves as a logical container for related objects such as tables and views. Each database contains zero or more tables, which contain data organized into rows and columns. Similarly, each table consists of columns that store data and indexes that improve query performance. Finally, views are virtual tables that combine the results of one or more underlying tables, similar to SQL joins.
         
         Below is an example diagram showing the relationships among the components described above:
         
         
         
         
         
         Fig. 2 - Example Impala Catalog Diagram
         
         
         
         # 3.关键技术实现过程
         ## 3.1 Overview
         
         Before implementing authentication and authorization features in Impala, let's go over how impala daemons interact with the client application and the various communication protocols involved in this interaction. Impala Daemon interacts with Client Application via Thrift RPC interface, which handles requests from the CLI, ODBC driver and JDBC driver. RPC calls are then routed to Impala Server where actual processing happens. Impala Server sends responses back to the respective clients based on the requested operation type. 
        
         
         
         During normal execution flow of a query, Impala Daemon communicates with Impala Server, which processes the incoming request. First, the Daemon sends a TQueryPlanRequest message to Impala Server requesting a plan for the given query statement. After receiving the PlanResponse message from Server, the Daemon generates and executes the Query Execution Plan. Each node in the plan runs its corresponding ExecTask code to compute the result set of the operation performed by the operator. 
         
         
         
         The Executor threads generated by each op run independently in parallel to generate partial results, which are then aggregated together by the Coordinator thread of the plan. Finally, the final result set is sent back to the Client Application.
         
         
         
         To implement authentication and authorization features in Impala, we'll add an extra layer of indirection between Impala Daemon and Impala Server. We'll introduce a new component called Authenticator which will handle the verification and validation of the user's credentials passed along with each request to Impala Server. Upon successful authentication, Authenticator will extract the authorized roles for the user and pass them along to Impala Server along with the original request.
         
         
         
         The detailed steps involved in the authentication and authorization implementation are discussed below. 

         ## 3.2 New Components
         
         ### 3.2.1 Authenticator Component
         
         The Authenticator component is responsible for validating the user's credentials and extracting the list of authorized roles for that user. We'll start by modifying the Impala thrift API to include fields for specifying the user name and password when initiating a connection to Impala Daemon. 

         
         ```diff
         + /**
         + * Specify the user name and password for connecting to Impala.
         + */
         + struct TImpalaDaemonAuthParams {
         +   1: required string user; // Required field for specifying the user name.
         +   2: optional binary password; // Optional field for specifying the password. Default empty string is assumed if no value provided.
         + }

         /**
          * Used to initiate connections to Impala and issue queries. Clients can make multiple concurrent
          * requests to the same Impalad. Multiple requests can be pipelined to improve overall query
          * performance. Clients should retry failed requests due to errors reported by Impalads.
          */
         service ImpalaService {
            ...

             /**
              * Connect to an Impalad instance and optionally authenticate. The returned Connection ID is used
              * to identify the session when executing queries. If 'user' and 'password' are specified, the client
              * must successfully complete the handshake by calling Authenticate() before issuing any queries.
              */
             i32 OpenSession(1:TOpenSessionReq req);

             void CloseSession(1:i32 sessionHandle);

            +/**
            + * Send user name and password during initial handshake to validate and authorize subsequent requests.
            + */
            +void AuthenticateUser(1:i32 sessionId, 2:string username, 3:binary password);
         }

         ```

         Next, we'll modify the ImpalaServer class to accept and parse the user name and password parameters passed during authentication. We'll also check whether the user exists in the backend auth provider and if the password matches the expected hash values stored in the backend.

         
         ```cpp
         Status ImpalaServer::AuthenticateUser(const string& username, const string& password) {
             DCHECK(!auth_provider_.get());
             ScopedSpinLock l(&auth_lock_);
             auth_provider_.reset(new AuthProvider);
             bool success = auth_provider_->InitBackend();
             if (!success) return Status("Unable to initialize backend");

             Status status = auth_provider_->VerifyPasswordHash(username, password);
             if (!status.ok()) return status;

             string principal = Substitute("$0/$1", impala_conf_.hostname(), username);
             scoped_refptr<AuthzToken> token(new AuthzToken(principal));
             AddAuthorizedRoles(*token);
             VLOG(2) << username << " connected.";
             return Status::OK();
         }
         ```

         The InitBackend method initializes the backend authentication provider depending upon the configuration. Currently, there is only support for Apache Ranger based authentication provider. If the initialization fails, we'll report an error message and exit. Otherwise, we'll call the VerifyPasswordHash function of the backend provider to verify the user's credentials and retrieve their hashed password value.

         
         Then, we'll modify the HandleRpcs() method to dispatch the AuthenticateUser function whenever a user tries to connect to Impala Daemon.


         ```cpp
         void ImpalaServer::HandleRpcs() {
           ...
            rpc::ThriftServer* processor =
                new rpc::ThriftServer(this, server_address_, num_acceptors_, port_);
            processor->setProtocolFactory(protocol_factory_);


            // Start listening for incoming requests
            processor->serve();
            Shutdown();
            VLOG(1) << "RPC server started";
            AuthenticateUser("admin", "");  //TODO : Remove hardcoding admin account once Authn enabled.

         }
         ```

         The last step is to implement the CreateSession() function of the ImpalaServer class to extract the user name and password from the thrift structs and send them to the AuthenticateUser function. 


         ```cpp
         Status ImpalaServer::CreateSession(const TCreateSessionReq& req,
               TCreateSessionResp* resp) {
            const string& username = req.impala_daemon_auth_params.user;
            const string& password = "";  // TODO : Get password from thrift structure.
            if (req.impala_daemon_auth_params.__isset.password) {
                 password = BinaryToHexString(req.impala_daemon_auth_params.password);
            }

            Status status = AuthenticateUser(username, password);
            if (!status.ok()) {
                LOG(ERROR) << "Error authorizing user '" << username
                    << "' for session creation: " << status.GetErrorMsg();
                return status;
            }

           // Rest of the logic remains unchanged.
         }
         ```

         Now, the ImpalaDaemons can receive user names and passwords during initial handshake and forward them to the Impala Server for authentication. 

        ### 3.2.2 Catalog Service Integration

        We've added the necessary pieces to handle user authentication and authorization. However, as per current design, the Catalog Service doesn't yet know anything about the user roles and hence won't restrict the access of non-authorized users. We'll modify the Catalog Service to track the user roles and apply access restrictions accordingly.


        First, we'll change the behavior of the AuthorizeSystemTablePrivilege() function of the Catalog class to return true always. This way, even if the table is marked as private, any authorized user would be able to view its schema and metadata. We don't want to block the ability of authorized users to explore such tables just because they haven't been granted access to their own data.

         
        ```cpp
        bool Catalog::AuthorizeSystemTablePrivilege(const string& db_name,
                                                     const string& table_name,
                                                     const TAuthorizationOperation operation) {
            return true;
        }
        ```

        Next, we'll modify the MetadataOpExecutor::Exec() function of the ExecEnv to check for the user roles before authorizing the access to the requested metadata.

        ```cpp
        Status MetadataOpExecutor::Exec(THandleIdentifierDesc desc,
                                        const TMetadataOpcode opcode,
                                        const TGetTablesReq& req,
                                        TGetTablesResp* resp) {
            if (!check_access_) return Status::OK();
            string db_name, tbl_name;
            RETURN_IF_ERROR(opcode_to_db_table_names_[opcode](desc, &db_name, &tbl_name));
            vector<TResultRow> result;

            const RequestPool* pool = request_pool_;
            if (opcode!= TMetadataOpcode::GET_TABLES &&!pool->has_role("all")) {
                const TQueryExecRequest* exec_request = nullptr;
                if (exec_requests_.count(desc)) {
                    exec_request = exec_requests_[desc].back().get();
                } else {
                    lock_guard<mutex> l(lock_);
                    auto entry = prepared_statements_.find(desc);
                    if (entry == prepared_statements_.end()) {
                        return Status::OK();
                    }
                    exec_request = entry->second->query_exec_request();
                }

                if (!exec_request ||
                   !pool->HasAccess(exec_request->stmt,
                                     exec_request->stmt_type,
                                     exec_request->authorization_request())) {
                    return Status::OK();
                }
            }

            switch (opcode) {
            case TMetadataOpcode::GET_SCHEMAS: {
                vector<TTableName> tables;
                RETURN_IF_ERROR(catalog_->GetAllDatabaseNames(tables));
                for (const TTableName& tname : tables) {
                    TTableRowResult row;
                    row.__set_columns({tname});
                    result.push_back(row);
                }
                break;
            }
            case TMetadataOpcode::GET_TABLES: {
                ResultSchema schema =
                    MetadataOpExecutor::MakeResultSchema({SQL_TYPE_STRING}, {"table_name"});
                const Database* db = nullptr;
                RETURN_IF_ERROR(catalog_->GetDb(db_name, &db));
                vector<string> table_names;
                RETURN_IF_ERROR(db->GetAllTableNames(&table_names));
                for (const string& table_name : table_names) {
                    TTableRowResult row;
                    row.__set_columns({{table_name}});
                    result.push_back(row);
                }
                break;
            }
            case TMetadataOpcode::GET_COLUMNS: {
                ResultSchema schema = MetadataOpExecutor::MakeResultSchema(
                    {SQL_TYPE_STRING, SQL_TYPE_STRING, SQL_TYPE_STRING,
                     SQL_TYPE_STRING, SQL_TYPE_STRING, SQL_TYPE_STRING},
                    {"table_name", "column_name", "data_type", "comment",
                     "is_key", "sort_order"});
                const Database* db = nullptr;
                RETURN_IF_ERROR(catalog_->GetDb(db_name, &db));
                const Table* table = nullptr;
                RETURN_IF_ERROR(db->GetTable(tbl_name, &table));
                vector<ColumnSchema> cols;
                RETURN_IF_ERROR(table->GetAllColumns(&cols));
                int sort_idx = 1;
                for (int i = 0; i < cols.size(); ++i) {
                    const ColumnSchema& col = cols[i];
                    string key_str;
                    if (col.is_key) {
                        key_str = "true";
                    } else {
                        key_str = "false";
                    }

                    TTableRowResult row;
                    row.__set_columns({{table_name,
                                       col.name,
                                       col.type.ToString(),
                                       col.comment,
                                       key_str,
                                       to_string(sort_idx)}});
                    result.push_back(row);
                    sort_idx++;
                }
                break;
            }
            case TMetadataOpcode::GET_FUNCTIONS: {
                ResultSchema schema = MetadataOpExecutor::MakeResultSchema(
                    {SQL_TYPE_STRING, SQL_TYPE_STRING, SQL_TYPE_STRING},
                    {"function_name", "return_type", "signature"});
                unordered_map<string, FunctionSignature> signatures;
                RETURN_IF_ERROR(catalog_->GetAllFunctions(&signatures));
                for (const pair<string, FunctionSignature>& item : signatures) {
                    if (item.first.find(".") == string::npos && item.first!= "*") continue;
                    if (!(db_name.empty() ||
                          boost::algorithm::starts_with(item.first, db_name + "."))) continue;
                    if (!CatalogUtil::MatchesPattern(item.first,
                                                      req.schema_pattern, req.table_pattern)) {
                        continue;
                    }
                    TTableRowResult row;
                    row.__set_columns({item.first,
                                      item.second.result_type.ToString(),
                                      item.second.ToString()});
                    result.push_back(row);
                }
                break;
            }
            case TMetadataOpcode::INVALID:
                break;
            }

            *resp = TGetTablesResp(move(result));
            return Status::OK();
        }
        ```

        The modified function now adds an additional check to see if the authorized user has access to the requested metadata. If the authorized user hasn't been granted access, the function simply returns OK and doesn't populate the response. 

        Lastly, we'll modify the ShowFunctions() and DescribeFunction() functions of the FE to incorporate the user roles check before returning the metadata of system functions.

        ```cpp
        Status Catalog::ShowFunctions(const std::string& db_name,
                                      const std::vector<std::string>& fn_names,
                                      const bool show_aggregate,
                                      const std::string& match,
                                      const std::string& user,
                                      std::vector<std::shared_ptr<TFunctionDesc>>* result) {
            Lock guard(lock_);
            const Database* db = FindDbOrDryRun(db_name);
            if (db == nullptr) return Status::OK();

            for (const string& fn_name : fn_names) {
                const Function* func = db->FindFunction(fn_name);
                if ((show_aggregate && func->is_aggregate()) ||
                    (!show_aggregate &&!func->is_aggregate())) {
                    if ((!match.empty() &&
                        !fn_name.compare(0, match.length(), match) &&
                         fn_name.size() > match.length()) ||
                            fn_name == "*" ||
                            fn_name == "%") {

                        // Check privilege for function
                        TAuthorizationOperation op = TAuthorizationOperation::SHOW_FUNCTION;
                        if (user.empty() ||!CheckAccess(db, "", fn_name, op)) {
                            LOG(INFO) << "Unauthorized user " << user << " trying to access function "
                              << db_name << "." << fn_name;
                            continue;
                        }

                        std::shared_ptr<TFunctionDesc> fn_desc(new TFunctionDesc());
                        fn_desc->__set_name(db_name + "." + fn_name);
                        fn_desc->__set_signature(func->signature().ToString());
                        fn_desc->__set_aggregate(func->is_aggregate());
                        (*result).emplace_back(std::move(fn_desc));
                    }
                }
            }
            return Status::OK();
        }

        Status Catalog::DescribeFunction(const std::string& db_name,
                                         const std::string& fn_name,
                                         const TFunctionFileType::type type,
                                         const std::string& user,
                                         std::string* signature,
                                         std::string* ret_type,
                                         std::vector<TColumn>* arg_types,
                                         bool* is_aggregate) {
            Lock guard(lock_);
            const Database* db = FindDbOrDryRun(db_name);
            if (db == nullptr) return Status::NotFound("Database does not exist.");

            const Function* func = db->FindFunction(fn_name);
            if (func == nullptr) return Status::NotFound("Function does not exist.");

            // Check privilege for function
            TAuthorizationOperation op = TAuthorizationOperation::DESCRIBE_FUNCTION;
            if (user.empty() ||!CheckAccess(db, "", fn_name, op)) {
                LOG(INFO) << "Unauthorized user " << user << " trying to describe function "
                  << db_name << "." << fn_name;
                return Status::OK();
            }

            *ret_type = func->signature().result_type.ToString();
            *signature = func->signature().ToString();
            *is_aggregate = func->is_aggregate();
            size_t num_args = func->signature().argument_types.size();
            arg_types->resize(num_args);
            for (int i = 0; i < num_args; ++i) {
                arg_types->at(i).__set_col_name("");
                arg_types->at(i).__set_col_type(func->signature().argument_types[i].ToString());
                arg_types->at(i).__set_is_partition_col(false);
            }
            return Status::OK();
        }
        ```

        The modified functions now add another condition to check the user's privileges before populating the output vectors.