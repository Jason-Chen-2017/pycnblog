
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Contributing to open-source projects on GitHub is a great way to learn and practice technical skills, get involved in a community, and improve your programming proficiency. However, becoming an effective contributor can be challenging as it requires different skill sets across various areas such as coding, design, documentation, and communication. In this article, we will explore the core concepts and tools required for contributing effectively on GitHub, including issue tracking, branch management, pull requests, code reviews, and more. We will also cover practical aspects of creating meaningful contributions, including how to write quality issues and pull requests, use well-structured commit messages, follow best practices, and communicate effectively with others. Overall, our goal is to provide you with all the information necessary to become an effective contributor on GitHub.

By the end of this article, you should have a solid understanding of the basic concepts, processes, and tools used for effective contribution on GitHub. You'll be able to start making meaningful contributions, both small and large, by following best practices. And having read through this article, you should feel confident in your ability to make valuable contributions to any open source project on GitHub!

# 2.基础概念与术语
Before we dive into the details of contributing to open-source software on GitHub, let's go over some basic concepts and terminology that will help us navigate the vast world of open-source development.

## 2.1 Issues
Issues are a fundamental part of any collaborative process on GitHub. They allow developers to report bugs or suggest improvements, ask questions, or share ideas related to a particular project. When someone creates a new issue, they typically include a descriptive title, a summary of the problem or enhancement, as well as any relevant links or attachments. The content of the issue may then be discussed and addressed by other contributors, who add comments and/or suggestions. Once the conversation has converged on a solution, one person may close the issue using their decision. 

## 2.2 Forks & Clones
Forking is the act of copying a repository from another user's account onto your own account. This creates a personal copy of the original repository, where you can work on it without affecting the original. Forks are often used when you want to propose changes to a project but don't have permission to directly push your changes back to the main codebase. For example, if you wanted to contribute to a popular JavaScript library, you might fork the repository, make your changes, and submit a pull request back to the original author.

Cloning is the act of downloading a local copy of a repository onto your computer. By cloning a repository, you can create a working environment on your machine and work on the code locally. Cloning repositories allows you to test out changes before submitting them back to the original repository. It's important to note that you shouldn't clone public repositories unless you intend to contribute to the project, as doing so could potentially put you at risk of violating the terms of its license agreement. If you need to keep private code safe, consider using version control services like Gitlab or Bitbucket instead. 

## 2.3 Branches
Branches are parallel versions of a repository that developers can work on independently. Each branch acts as a sandbox where you can experiment with features, fix bugs, or otherwise try things out without disrupting the main codebase. Branch names should be short and descriptive, and should not contain special characters or spaces. When you're ready to merge your changes back into the main codebase, you can do so using a pull request.

## 2.4 Commits
Commits are snapshots of the state of the repository at a given time. Every time you save changes to your code, you must commit these changes alongside a message describing what was changed. Good commit messages make it easier for other developers to understand why certain changes were made, which helps maintainability and collaboration. It's essential to write clear and concise commit messages that explain what each change does and why it was made.

## 2.5 Pull Requests (PRs)
Pull requests (PRs) are the mechanism by which developers submit changes to a project's codebase. When a developer wants to propose a set of changes to a repository, they create a PR against the target branch (usually "master"). Other contributors review the proposed changes and leave comments, indicating whether there are any issues or requested changes. After approval, the changes can be merged into the master branch and included in the next release of the project.

It's crucial to carefully review PRs to ensure that only high-quality code makes it into the main codebase. Developers should also pay attention to testing and security procedures that may exist within the repository, as well as adhere to any style guidelines or standards set forth by the organization. 

## 2.6 Code Reviews
Code reviews are checks and balances that take place after a set of changes have been submitted via a pull request. These reviews involve reviewing the actual code itself for correctness, ensuring that it follows established conventions and styles, and identifying any potential issues or vulnerabilities. During code reviews, developers may flag up specific lines of code that they'd like reviewers to look at, as well as suggest alternative approaches or implementations.

In conclusion, knowing the basics of how to contribute effectively on GitHub involves familiarizing yourself with key concepts and technologies such as issues, branches, commits, and pull requests. Additionally, you should strive to write clear and comprehensive commit messages that detail every single line of code that was added, modified, or removed. Finally, being mindful of code reviews during submission process is essential to ensuring that the overall code quality remains high throughout the entire lifecycle of the project.