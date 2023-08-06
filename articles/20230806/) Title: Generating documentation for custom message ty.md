
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         In this article I will explain how to generate documentations of the custom message types created by ROS users and developers. Documentation is one of the most important aspects of any open-source project as it helps other contributors understand the code better and contribute more effectively. It also provides a high level view on what the software does and how it works for those who do not have access to the source code or want an easier way to navigate through the code base. Therefore, having proper documentation is crucial for maintaining and developing ROS packages. 
         
        To create good documentation, we need to follow some key principles such as clarity, accuracy, consistency, and completeness. Good documentation should include all necessary information about the package including installation steps, basic usage examples, API reference, troubleshooting guides, and contribution guidelines. Properly generating documentation can help improve communication between different members of a team, make it easier for newcomers to join your project, and ultimately increase the overall quality of your work. 

         In order to automate the process of creating documentation for custom message types in ROS, there are several tools available that can be used depending on the programming language used. Some popular options include Doxygen (C++), Epydoc (Python), Sphinx (Python/Restructured Text) etc. Here I will focus only on explaining how to use Python's Sphinx library to automatically generate documentation for custom messages written in Python using ROS.

         # 2. Basic concepts and terminology
         Before proceeding with the actual documentation generation process, let’s first go over some fundamental terms and concepts related to documentation. 
         
         ### Packages
         A package in ROS is basically a collection of nodes, scripts, configurations, launch files, CMakeLists.txt, msg, srv, action files, etc. All these files together form a single unit which makes up a complete robotic system. The manifest file describes each package along with its name, version number, maintainer details, dependencies, description, website URL, licenses, and so on. Each package has unique identifier called the package name which consists of three parts separated by slashes (i.e., <group_name>/<package_name>). For example, sensor_msgs package belongs to the built-in ROS messages group while a turtlesim package is part of the TurtleBot Robotics Simulator community.

          ### Messages and Services
          While publishing data, clients typically subscribe to topics that provide them the type of data they require. Similarly, services offer clients a way to interact with the nodes on the server side without needing to know their implementation details. Both of these entities are defined in the form of messages and therefore have corresponding schemas containing fields and structures defining the data being exchanged. There are two main categories of messages - simple messages and complex messages. Simple messages contain scalar values like integers, floats, bool, strings, while complex messages consist of arrays, nested structures, and other message types. Examples of both simple and complex messages include std_msgs/String, geometry_msgs/PoseStamped, sensor_msgs/LaserScan, nav_msgs/Odometry, etc.

          ### Message Definition File
          Every message type has a corresponding definition file stored in the'msg' folder of the respective package. This file contains information about the message structure, field names, default values, maximum and minimum limits, enumerations, and additional metadata required by various functions. The definitions provided in these files are used to automatically generate documentation for the messages during build time.

          ### Source Code Comments
          As mentioned earlier, comments within the source code itself play a vital role in understanding and navigating the code base. They provide clear descriptions of the purpose of individual blocks of code and can even serve as standalone tutorials for those who are unfamiliar with certain modules. Good commenting practices ensure that comments remain accurate, relevant, and useful throughout the codebase.

          # 3. Algorithmic Principles
         Now, let us move onto the core algorithmic principle behind automatic documentation generation for custom message types in ROS. We will use the Python language and the Sphinx library to demonstrate the procedure.

         ## Step 1: Create a Custom Message Type
         First step would be to define our own custom message type and place it in a separate package. Let's say we have created a custom message type called "CustomMsg" and placed it in the custom_message package under the my_robot namespace. We can create a definition file for our message using the following syntax:

      ```python
      uint32 id           # Unique ID for object
      string name        # Name of the object
      float64 weight     # Weight of the object in kg
      ```

       Save this file as `custom_msg.msg` in the custom_message package's `/msg` directory.
       Additionally, you may add additional attributes as per requirements. Once done, create a python module inside the same package called `custom_msg.py`. Add the following import statement at the beginning of the file:

      ```python
      from custom_message.msg import custom_msg 
      ```
      
      You can now start writing publisher, subscriber and service node(s) using the above message type in your ROS program. Publishers, subscribers, and services can then publish and receive data of type custom_msg respectively. Here is an example of what the publisher node might look like:

      ```python
      #!/usr/bin/env python
      import rospy
      from custom_message.msg import custom_msg
  
      def pub():
          pub = rospy.Publisher('mytopic', custom_msg, queue_size=10)
          rospy.init_node('publisher_node', anonymous=True)
          rate = rospy.Rate(10) # 10hz
          i = 0
          while not rospy.is_shutdown():
              msg = custom_msg()
              msg.id = i
              msg.name = "Object_" + str(i)
              msg.weight = round((i+1)*0.5,2) 
              pub.publish(msg)
              print("Published : ", msg)
              i += 1
              rate.sleep()
  
      if __name__ == '__main__':
          try:
              pub()
          except rospy.ROSInterruptException:
              pass
      ```

      Note that we imported the custom_msg type from the custom_message package before declaring variables of that type and passing them to the publisher node. Subscribers and services would work similarly.

   ## Step 2: Generate Documentation Using Python's Sphinx Library
   Next, we will install the sphinx package and use it to generate documentation for our custom_message package. Follow the instructions below to set up Sphinx and generate documentation for your custom message type:
   
   1. Install Sphinx: If you don't already have Sphinx installed on your computer, run the following command:
   
   ```
   pip install sphinx
   ```
   
    2. Clone the ROS source repository: You need to clone the entire ROS source repository into a local machine so that you can browse around the existing ROS packages and get familiar with their structure and contents.
   
    3. Navigate to the location where you cloned the ROS repo. Browse to the custom_message package and locate the msg subdirectory. Inside the msg subdirectory, you should find the custom_msg.msg file you just created earlier.
   
    4. Open a terminal window and navigate to the custom_message package directory. Then execute the following commands to generate the initial configuration files for building documentation:
   
   ```
   cd docs
   mkdir source _build
   sphinx-quickstart
   ```
    
    5. Follow the prompts shown by the sphinx-quickstart utility and select appropriate settings based on your preferences. Make sure to enable "autodoc" extension when asked to choose extensions. Once completed, you should see a new set of configuration files named "conf.py", "index.rst", and others inside the "docs" directory.

    6. Edit the conf.py file by adding the following lines at the end of the file:

   ```
   import os
   import sys
   import subprocess
   
   # Import the custom message package after importing sphinx since it 
   # relies on it internally.
   from custom_message.msg import custom_msg 
   
   sys.path.insert(0, os.path.abspath('../../src'))
   
   # -- General configuration -----------------------------------------------------
   
   master_doc = 'index'
   project = u'my_robot'
   copyright = u'2021, MyRobot Developers'
   author = u'MyRobot Developers'
   
   release = subprocess.check_output(['git', 'describe']).strip().decode('utf-8')
   
   # -- Extensions configuration ---------------------------------------------------
   
   extensions = [
     'sphinx.ext.autodoc',
     'sphinx.ext.napoleon',
   ]
   
   autodoc_default_options = {
     'members': True,
      'undoc-members': True,
     'show-inheritance': True,
   }
   
   templates_path = ['_templates']
   exclude_patterns = []
   
   napoleon_use_ivar = True
   
   # -- Options for HTML output ---------------------------------------------------
   
   html_theme = 'alabaster'
   html_static_path = ['_static']
   htmlhelp_basename ='my_robotdoc'
   ```

   7. At this point, the documentation setup should be ready. Run the following command inside the docs directory to build the documentation:
   
   ```
   make html
   ```
   
   The generated documentation will be saved inside "_build" directory in the "html" subfolder. Open the index.html file located inside the "_build/html" directory to view the generated documentation.

   By default, the generated documentation includes a summary table showing the list of all the modules, classes, functions, and attributes present in the given package. You can further customize the documentation by editing the rst files located inside the "source" directory. 

   9. Finally, once you're satisfied with the final state of your documentation, commit your changes back to the remote repository. Github pages can be configured to host your documentation online if needed.

   # Conclusion
   In this article, we discussed automated documentation generation for custom message types in ROS using Python's Sphinx library. We demonstrated how to create a custom message type, configure Sphinx, and generate documentation for our custom message type using standard techniques like comments and docstrings. Overall, this tutorial aims to teach readers how to approach the problem of auto-generating documentation for custom message types using tools like Sphinx and the knowledge gained from studying ROS and Python fundamentals.