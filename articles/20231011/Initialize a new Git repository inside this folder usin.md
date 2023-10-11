
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## What is GIT?
Git is an open-source version control system (VCS) used for tracking changes in computer files and coordinating work on those files among multiple people. It was initially designed to handle source code but has since become the most widely-used VCS in modern software development.[3]
## Why use GIT?
Git offers several advantages over other VCS systems such as Subversion or Perforce:

1. Simple branching model – Many developers may be working on different features simultaneously which can make it difficult to maintain a consistent set of files across all branches. With Git, you create a separate branch for each feature you are working on, which makes it easier to isolate your changes without affecting other developers’ work.

2. Distributed architecture – Unlike centralized servers like CVS or SVN that have one single point of failure, Git does not rely on a single server to store all data. Instead, Git stores a copy of every commit along with its parent(s). This allows for faster clones and backups than other VCS systems while still allowing full access to all history and commits.

3. Efficient merging – When two or more developers edit the same file concurrently, conflicts arise when they try to merge their changes back together. In contrast, Git automatically resolves any conflicts during the merging process. Additionally, Git detects when lines were moved or deleted during rebasing and only applies those changes to the current branch rather than throwing them away completely.

4. Well-organized workflow – The nature of Git encourages well-organized workflows where small, atomic commits are made frequently and thoroughly tested before they are merged into the main codebase. Furthermore, Git provides powerful tools like rebase, stash, bisect, and reflog to help developers manage their work flow and reduce errors.

In summary, by using Git, you can track changes in complex codebases more easily and collaborate effectively with other developers. Its ability to provide lightweight yet robust backup functionality, speedy cloning and fast merging also make it an excellent choice for large scale projects requiring efficient collaboration and versioning.
## How Does GIT Work?
To understand how Git works, let's take a look at the following high-level overview diagram:


1. Working Directory - This is the local area where you will do most of your editing. Files here are not tracked by Git until they are committed. You can think of this directory as a snapshot of what the project looks like on disk right now.

2. Local Repository - This is the.git folder located within the root directory of your project. It contains everything related to the Git history of your project including information about the state of your working tree, metadata about individual commits, references to remote repositories, etc. If you delete this folder, you lose your entire Git history.

3. Staging Area - This is another important concept in Git. It represents a collection of modified files that will be included in the next commit. Any changes that are added to the staging area must pass certain checks to ensure that they conform to best practices for formatting and readability. These modifications don't actually change anything on disk just yet.

4. Commit - A commit is essentially a snapshot of the current state of the repository. Each time you perform a commit, you're taking a snapshot of the current status of the project and storing it permanently in the local repository. The commit message serves as documentation for why the changes were made and should explain any contextual details necessary.

5. Remote Repositories - These are typically hosted on services like GitHub, GitLab, Bitbucket, or self-hosted solutions like Gitlab CE or Satellite. They allow you to share your code with others and provide a central place to store your work. When you push a commit to a remote repository, it sends the updates from your local repository to the remote repository. Other users can then pull these updates down and integrate them into their own copies of the project.