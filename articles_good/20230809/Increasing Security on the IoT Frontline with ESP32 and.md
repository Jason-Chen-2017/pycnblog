
作者：禅与计算机程序设计艺术                    

# 1.简介
         

        The Internet of Things (IoT) has revolutionized our lives in many ways. However, it also brings new security risks to devices connected to this network. One such risk is eavesdropping attacks or man-in-the-middle attacks that can capture sensitive data transmitted over the internet. These attacks pose a serious threat to any system that relies heavily on secure communication channels for device authentication, access control, and data protection. 
        
        Secure Sockets Layer (SSL)/Transport Layer Security (TLS) protocol suites provide security standards for establishing encrypted connections between web servers and clients as well as between different application layers. While they are widely used in today's applications, their weaknesses have not been fully explored yet. As the world becomes more interconnected by these devices, these vulnerabilities become increasingly critical. In recent years, the use of these protocols continues to grow due to its low overhead compared to other alternatives like WPA2/WPA3. Therefore, the need arises to address these vulnerabilities effectively to keep up with the ever-growing IoT ecosystem.
        
       In this article, we will explore how the SSL/TLS handshake works and what changes were made to enhance the security of these protocols and enable them to work seamlessly with ESP32 microcontrollers. We will start by understanding the basic concepts behind TLS before moving on to explain the specific implementation details of ESP32 using TLS 1.3. Finally, we will conclude with an exploration into the future directions and challenges ahead of us.

        Let’s get started!

        
        
        # 2. Basic Concepts
        ## 2.1 SSL/TLS Handshake 
        Before diving into the details of implementing TLS 1.3 on the ESP32 platform, let’s understand the basics of SSL/TLS handshaking first. 

        When you visit a website secured by SSL/TLS, your browser sends out a request to connect to the server and initiate the SSL/TLS handshake process. Here are the steps involved:

       Client Hello - This message sent from client to server indicates which version of SSL/TLS protocol is being used. It also includes various encryption methods supported by both parties including cipher suite identification information. The list of cipher suites supported by each version of SSL/TLS differs slightly. 

       Server Hello - The server responds back with its chosen version of SSL/TLS along with certificate information. The server certificate contains the public key of the server, which allows the client to verify the identity of the server during the session. Certificate signing authority (CA) certificates are included in the response if the server uses HTTPS.

       Server Key Exchange - This step involves exchanging symmetric keys through a non-repudiation mechanism called key exchange algorithm. The algorithms used here include Diffie Hellman key exchange, RSA key exchange, or Elliptic Curve Diffie Hellman key exchange. 

       Client Key Exchange - After the shared secret key is established, the client generates the final hash value to be signed with the server’s private key. If the signature verifies successfully, the two sides of connection are authenticated and ready to communicate securely.

       Cipher Suite Negotiation - Once the SSL/TLS handshake is complete, the negotiated ciphersuite informs the peer about the encryption method being used for subsequent messages.

        ## 2.2 Introduction to TLS 1.3 
        TLS stands for Transport Layer Security and was introduced in 1999 as a replacement for SSL. It defines the rules for communicating securely over a computer network without being able to read the underlying data being transmitted. Over time, there have been several revisions of the protocol, some of which contain significant improvements over previous versions. TLS 1.3 was recently released by the IETF (Internet Engineering Task Force) and is expected to be deployed across most modern browsers, operating systems, and applications. 

        The main improvement brought by TLS 1.3 is the ability to establish multiple sessions within a single connection. This enables the client to establish multiple virtual links between the server and the client without requiring additional round trips. Another major feature is the support for PFS (perfect forward secrecy), which ensures that previously established connections cannot be decrypted even if the underlying key is compromised. 

        TLS 1.3 does away with several older features, making it easier to implement and less prone to security vulnerabilities. For example, the entire key exchange takes place over a mutually authenticated and encrypted channel instead of using separate static DH parameters for key exchange. Similarly, session resumption is built into the protocol itself instead of relying on session tickets. Overall, the TLS 1.3 protocol is considered highly secure and should make secure communications easy and efficient for developers.


        # 3. Implementation Details
        ## 3.1 Implementing TLS 1.3 on ESP32
        To implement TLS 1.3 on the ESP32 platform, we need to do the following tasks:

        * Set up a TCP socket connection to the remote server using ESP-IDF APIs.
        * Create a TLS context object using mbedtls library functions.
        * Configure the TLS context object to use TLS 1.3 version of the protocol.
        * Perform the SSL/TLS handshake with the remote server.
        * Send and receive data securely over the connection using the TLS context.

        ### Configuring mbedtls Library
        First, we need to download and install the latest version of the mbedtls library using the instructions provided in the official documentation.

        Next, we need to import the necessary header files and link against the mbedtls library. Open the project configuration file (idf.py menuconfig) and navigate to Component config > mbedTLS. Ensure that MBEDTLS_CONFIG_FILE option is set to “mbedtls/esp_config.h”. This will ensure that all the required settings related to mbedtls are automatically configured when building the firmware image.

        Next, we need to configure the TLS protocol version to be used by setting the minimum and maximum versions allowed in the ssl_conf struct. Add the below line after initializing the mbedtls library inside app_main():

           esp_err_t ret = mbedtls_ssl_config_defaults(&conf, 
                   MBEDTLS_SSL_IS_SERVER,
                   MBEDTLS_SSL_TRANSPORT_STREAM,
                   MBEDTLS_SSL_PRESET_DEFAULT);
           conf.min_version = MBEDTLS_SSL_VERSION_TLS1_3;
           conf.max_version = MBEDTLS_SSL_VERSION_TLS1_3;

            We also need to disable certain ciphersuites that may be vulnerable to attack surface area identified earlier. Specifically, add the lines below after configuring the minimum and maximum protocol versions:

           // Disabling DHE key exchange for increased security
           int index = 0;
           while(index < conf.ciphersuite_list[0]) {
               if(MBEDTLS_SSL_CIPHERSUITE_WITH_DHE_CAMELLIA_128_CBC_SHA256 == conf.ciphersuite_list[index] ||
                  MBEDTLS_SSL_CIPHERSUITE_WITH_DHE_CAMELLIA_256_CBC_SHA256 == conf.ciphersuite_list[index] ||
                  MBEDTLS_SSL_CIPHERSUITE_WITH_DHE_RSA_AES_128_GCM_SHA256 == conf.ciphersuite_list[index] ||
                  MBEDTLS_SSL_CIPHERSUITE_WITH_DHE_RSA_AES_256_GCM_SHA384 == conf.ciphersuite_list[index] ||
                  MBEDTLS_SSL_CIPHERSUITE_WITH_DHE_RSA_CHACHA20_POLY1305_SHA256 == conf.ciphersuite_list[index] ) 
               {
                   memset(&conf.ciphersuite_list[index], 0, sizeof(int));
                   continue;
               } else {
                   index++;
               }
           }

           // Setting preferred curves to P-256 for improved performance
           mbedtls_ecp_group_id grp_id[] = { MBEDTLS_ECP_DP_SECP256R1 };
           mbedtls_ssl_conf_curves( &conf, grp_id, sizeof(grp_id) );

           // Applying modified configurations
           mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_NONE);
           mbedtls_ssl_conf_rng(&conf, esp_random, NULL);
           mbedtls_ssl_conf_dbg(&conf, debug_callback, stdout);
           mbedtls_ssl_init(&ssl);
           mbedtls_x509_crt_init(&cacert);
           mbedtls_x509_crt_init(&clicert);
           mbedtls_pk_init(&pkey);
           mbedtls_entropy_init(&entropy);
           mbedtls_ctr_drbg_init(&ctr_drbg);
           if((ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy,
                              const_cast<unsigned char *>(seed), strlen(seed)))!= 0){
               printf("Failed to seed the random number generator!\n");
               goto exit;
           }
           if ((ret = mbedtls_ssl_setup(&ssl, &conf))!= 0) {
               printf("Failed to setup SSL context!\n");
               goto exit;
           }



        ### Implementing SSL/TLS Handshake
        Now that we have initialized the SSL/TLS context object, we need to perform the SSL/TLS handshake with the remote server. Here is the code snippet to accomplish this task:

           esp_err_t ret = mbedtls_net_connect(&server_fd, "example.com", port, 
                           MBEDTLS_NET_PROTO_TCP);
           if(ret!= 0){
               printf("Error connecting to %s:%d!\n", host, port);
               return;
           }
           printf("Connected to %s:%d\n", host, port);

           mbedtls_ssl_set_bio(&ssl, &server_fd, mbedtls_net_send, mbedtls_net_recv, NULL);
           
           while ((ret = mbedtls_ssl_handshake(&ssl))!= 0) {
               if (ret!= MBEDTLS_ERR_SSL_WANT_READ && ret!= MBEDTLS_ERR_SSL_WANT_WRITE) {
                   printf("Error - SSL handshake failed!\n");
                   break;
               }
           }

           if (ret!= 0) {
               printf("SSL/TLS connection failed : %d\n", ret);
               goto cleanup;
           }
           

           printf("SSL/TLS connection successful.\n");

           unsigned char buf[1024];
           size_t len;

           /* Read resource */
           memset(buf, 0, sizeof(buf));
           len = sizeof(buf)-1;
           ret = mbedtls_ssl_read(&ssl, buf, len);
           if(ret <= 0){
               printf("mbedtls_ssl_read returned -%#x\n", -ret);
               goto cleanup;
           }
           buf[len] = '\0';
           printf("%s\n", (char *)buf);


           /* Write resource */
           memset(buf, 0, sizeof(buf));
           snprintf((char*)buf, sizeof(buf)-1, "GET / HTTP/1.1\r\nHost: www.%s\r\nConnection: close\r\n\r\n", domain);
           len = strlen((const char*)buf);
           ret = mbedtls_ssl_write(&ssl, buf, len);
           if(ret <= 0){
               printf("mbedtls_ssl_write returned -%#x\n", -ret);
               goto cleanup;
           }
           printf("Request sent: GET / HTTP/1.1\r\nHost: www.%s\r\nConnection: close\r\n\r\n", domain);

           /* Read response */
           memset(buf, 0, sizeof(buf));
           len = sizeof(buf)-1;
           do {
               ret = mbedtls_ssl_read(&ssl, buf, len);
               if(ret == MBEDTLS_ERR_SSL_TIMEOUT){
                   /* Wait for timeout if blocking read */
                   printf("Timeout occurred\n");
               } else if(ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY){
                   /* Connection closed cleanly */
                   break;
               } else if(ret < 0){
                   printf("mbedtls_ssl_read returned -%#x\n", -ret);
                   goto cleanup;
               }else if(ret == 0){
                   printf("\nEOF\n");
                   break;
               }else{
                   len -= ret;
                   printf("%.*s", ret, (char*)buf);
                   fflush(stdout);
               }
           } while(1);

           /* Done, close the connection */
           printf("\nClosing the connection...\n");
   cleanup:
       mbedtls_net_free(&server_fd);
       mbedtls_x509_crt_free(&cacert);
       mbedtls_x509_crt_free(&clicert);
       mbedtls_pk_free(&pkey);
       mbedtls_ssl_free(&ssl);
       mbedtls_ssl_config_free(&conf);
       mbedtls_ctr_drbg_free(&ctr_drbg);
       mbedtls_entropy_free(&entropy);