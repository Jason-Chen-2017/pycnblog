
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Packet sniffing (抓包) is the process of capturing network traffic for analysis or debugging purposes. Packet sniffer programs can be used to gather information about packets being transmitted across a network or between two systems. They are commonly used in troubleshooting connectivity issues, intrusion detection, performance monitoring, and packet analysis.

This article will cover how to configure and use a popular open-source packet sniffer tool called Wireshark on an Ubuntu Linux system. Wireshark allows users to capture data from local area networks, wide area networks (e.g., internet), and tunnels such as Virtual Private Networks (VPNs). It provides powerful filtering capabilities that allow you to focus on specific types of traffic or network events. It also offers various visualization features, including network graphs, protocol histograms, and ASCII dumps of captured data.

To follow this tutorial, you should have basic understanding of networking concepts like IP addresses, subnetting, port numbers, etc. You should also have a working knowledge of command line tools and text editors. We assume you're using an up-to-date version of Ubuntu Linux and have at least some experience with Wireshark's GUI interface. 

We will start by installing Wireshark on our Ubuntu machine and exploring its basic functionality through the CLI. Then we'll switch over to the graphical user interface to perform more advanced tasks and customize the display. Finally, we'll learn how to write filters to filter out specific types of traffic and save them for later reference. 

# 2.Core Concepts and Relationships
Before we dive into configuring and using Wireshark, let’s briefly review some core networking concepts and relationships:

1. Networking Basics - In order to understand the role of packet sniffers, it's important to first understand what constitutes a network. A network is made up of two or more devices connected together via cables, wires, or fiber optic cables. Each device has one or more interfaces, which are physical connectors that transmit and receive digital signals. The protocols that each interface uses determine how the data gets sent around the network. Some common protocols include Ethernet, Wi-Fi, LAN/WAN, and VPN. 

2. IP Addresses and Subnetting - An Internet Protocol address (IP Address) is assigned to each device on a network. These addresses are typically expressed in dot notation, where each octet represents a number between 0 and 255. The last octet indicates the individual device within a subnet, while the second to third octets indicate the group of devices that share a common purpose or function. For example, the IP address 192.168.1.1 belongs to a home router, while 172.16.0.1 might belong to a printer server. When a device connects to another device on the same subnet, it can communicate directly without having to go through any intermediaries. If the devices need to communicate over a different subnet (for example, connecting to your laptop when remote accessing a company network), then the data must pass through routers or gateways that forward the data accordingly. This makes subrouting very critical for proper communication between devices on separate subnets. 

3. Port Numbers - Ports are logical connections inside computers or servers that establish communication between applications. Each application running on a computer requires a unique port number so that other devices can connect to it. By convention, ports below 1024 are reserved for privileged applications and the default port is usually TCP port 80 for web traffic. Different services may require different port numbers depending on their design and implementation. For example, if you run multiple instances of Apache on a single server, they could potentially conflict with each other unless they use different port numbers. 

4. Firewalls - A firewall is a security measure that monitors incoming and outgoing network traffic based on predefined rules or policies. It blocks unauthorized access to network resources and protects against malicious attacks. Common firewall technologies include iptables, nftables, and IPsec. A packet sniffer relies heavily on network security measures implemented by firewalls. Therefore, it's essential to ensure that firewalls are properly configured before attempting to use a packet sniffer. 

# 3.Algorithm and Operations
Now let's discuss how to set up Wireshark and use it effectively to analyze network traffic. Here's a high level overview of the steps involved:

1. Install Wireshark - Before we begin analyzing network traffic, we need to install Wireshark on our Ubuntu machine. Open the terminal and type: `sudo apt update && sudo apt upgrade` to get the latest package list. Next, type: `sudo apt install wireshark`. Wait until the installation completes.

2. Launch Wireshark - After installing Wireshark, launch it by typing: `wireshark` in the terminal. Alternatively, you can search for "Wireshark" in your application menu to launch it. You should see a window similar to the following screenshot:


3. Capture Traffic - To capture network traffic, we need to enable a network interface. On most Linux distributions, network interfaces are enabled automatically. However, if a new interface was added after installation, it may not be active yet. To check if your network interface is active, navigate to the Network Interfaces panel in Wireshark and click refresh. Scroll down to find your network adapter and make sure it says "UP." Otherwise, select the adapter and hit the green arrow icon to bring it UP. Once the interface is active, click the red circle button next to the Start button to begin capturing network traffic. Make sure to adjust the Filter dropdown box if necessary to only capture the desired traffic. Click Stop to stop capturing.

4. View Captured Data - Wireshark displays captured data in several tabs. The topmost tab is the Summary view, which shows a general summary of all captured traffic. Below that is the Details view, which gives detailed information about each packet. The Packet List tab lists every packet that was captured along with relevant metadata such as source and destination IP addresses, protocols, and timestamps. Finally, the Hex Dump tab allows you to examine the raw contents of each packet.

5. Customize Display - Wireshark comes with many options for customizing its display. From the preferences menu, you can change colors, themes, font sizes, and much more. Additionally, you can create custom columns in the Packet List tab to show additional data about each packet. Right-click on any column header and choose Add/Remove Columns to add or remove fields from the display. You can also zoom in and out of the captured data using the vertical scrollbar on the right side of the window.

6. Write Filters - Wireshark includes powerful filtering capabilities that allow you to isolate specific types of traffic. Navigate to the Statistics pane on the left side of the window and select the Protocol tab. Here, you can view a breakdown of packet counts by protocol. Double-click on any entry to apply a filter that captures only those packets. You can also edit existing filters or create new ones using the filter editor. Simply enter the criteria for the filter in the Expression field and press Enter. The filtered results appear in the Packet List tab with a yellow background indicating that they were filtered. To save a filter for future use, simply name it and click Save.

That's it! That's everything you need to know to use Wireshark effectively to capture and analyze network traffic. I hope this helps!