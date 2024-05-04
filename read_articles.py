if __name__ == '__main__':

    with open('articles_res.md', 'w') as resfile:

        with open('articles.md', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() != '':
                    print(f'"{line.strip()}",')
