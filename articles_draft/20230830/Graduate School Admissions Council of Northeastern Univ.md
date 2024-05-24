
作者：禅与计算机程序设计艺术                    

# 1.简介
  

本文将阐述 NSUGC 对计算机科学硕士生备选材料中英语语言要求的倡议。通过 NSUGC 的这一建议，使得计算机科学研究生在提出学术论文前能够更好地融入计算机科学研究领域。
计算机科学研究领域已经成为当今的重要技能之一。然而，英语水平并非所有计算机科学研究生都具备的基础要求。受到众多研究机构、教育机构和企业的需求，越来越多的高等院校开始对计算机科学研究生备选材料中英语语言要求作出明确规定。
近年来，随着研究生毕业率逐渐下降，计算机科学研究生入学的门槛也越来越低。因此，计算机科学研究生们不仅需要符合计算机科学相关专业的学习要求，还需要掌握英语语言能力。
如今，国内外许多高等院校都在鼓励计算机科学研究生提前熟悉英语，这对计算机科学研究生培养英语口语、阅读、表达等能力都有着不可替代的作用。通过 NSUGC 的这一建议，就能帮助计算机科学研究生进一步融入到计算机科学研究领域，提升个人竞争力和职业发展空间。
# 2.基本概念及术语
## 2.1 一般概念
### 2.1.1 研究生院（Department of Computer Science and Engineering）
研究生院，即计算机科学和工程系（Computer Science and Engineering Department），成立于1998年，是国内著名的计算机科学与信息技术学科的重要组成部分。其主要研究方向包括计算机系统结构、计算机网络、数据库系统、分布式计算、人工智能、机器学习、计算机图形学、云计算、大数据分析、数据中心网络、嵌入式系统等。该系主管单位是教育部计算机所。
### 2.1.2 计算机科学研究生
计算机科学研究生（Computer Science Master's degree program），是指在计算机科学或相关专业修读硕士学位的学生。通常情况下，计算机科学研究生拥有计算机科学或相关专业的学历，并且参加了相关课程，取得了优秀的综合素质总分。例如，一名计算机科学研究生在读期间获得了一定的成绩排名，通过教授的讲授能正确、深刻地理解知识点。同时，研究生应在积极参与相关实验室项目和比赛中锻炼自己的知识和能力。
## 2.2 专业术语
### 2.2.1 研究生导师
研究生导师（Assistant Professor），又称助教或外聘教授，是指在普通院校外开设学位课程给予教学指导的专业人员。一般为具有博士以上学位的学者担任，负责教授学生相关的专业课程。助教工作既可以由院系自行安排，也可以由教师自愿推荐。
### 2.2.2 普通外语
普通外语（General Education Program in Other Languages)，是指硕士阶段以外的其它国家语言学习或考试。普通外语学习可帮助学生在专业课堂上了解国际视野，增加国际化视野；普通外语学习还可提高语言学习能力，培养学生适应各种社会环境的能力；普通外语学习还可以增强学生的语言表达、组织能力及与他人的交流能力。
# 3.核心算法原理及具体操作步骤
NSUGC 主席陈琳琳表示："我非常赞同您的意见。过去几年来，计算机科学研究生备选材料中的英语语言要求越来越成为一种趋势。不少学校已经宣布计算机科学研究生必须要通过普通外语考试才能申请到资格。这无疑是减轻了计算机科学研究生负担，但也给予了计算机科学研究生更多的自主选择权。这是一个难得的契机，NSUGC 正是利用这种机遇，建议计算机科学研究生提前学习英语，以便更好地融入计算机科学研究领域。"
具体的操作步骤如下：
1. 通知：NSUGC 会收集各高校计算机科学研究生的意向，共同拟定新一轮的中文语言测试标准，通知研究生提交资料。
2. 提供材料：中文测试材料可根据学校需求提供。若没有相应材料，NSUGC 也会抽取在线视频或书籍作为材料，或组织相关课程的网络辅导。
3. 测试时间：前一轮的中文语言测试结束后，NSUGC 将发布考试时间表。在时间安排之前，研究生应充分准备。
4. 考试方式：由于英语水平的限制，目前一般只接受考试形式的中文语言测试。考试分为平时成绩考查、模拟试卷及口语考核三部分。平时成绩考查将侧重于理论学习，模拟试卷将考察语法、词汇、听力、阅读、写作等方面的技巧，口语考核将考察口头表达、口腔协调性、反应速度、翻译等方面。
5. 考试费用：目前，NSUGC 不收取任何考试费用。研究生应与辅导员取得联系，进行准备。
6. 考试结果：NSUGC 将发布各项考试的题库，并根据研究生实际情况对其进行调整。
# 4.具体代码实例及解释说明
举例来说，假设某一高校计算机科学研究生想提前学习英语，可以参考以下代码实现：

```python
def learning_english():
    print("Hi! I'm studying in the Department of Computer Science and Engineering at school XYZ.")
    
    while True:
        language = input("Would you like to learn english? (yes/no): ")
        
        if language == "yes":
            break
            
        elif language == "no":
            print("No problem, you can also come later when you are comfortable with your Chinese language test score!")
            return
            
        else:
            print("Invalid answer, please enter yes or no.")
        
    # 中文测试材料提供
    materials = ["Materials from high-school", "Materials from Internet"]

    selected_materials = random.choice(materials)
    print("I will use {} as my material.".format(selected_materials))

    # 测试时间安排
    year = time.strftime("%Y")
    month = int(time.strftime("%m")) - 1
    exam_date = datetime.datetime(year=year, month=(month+1), day=random.randint(1, calendar.monthrange(int(year), month)[1]))
    print("The next examination date is on {}, which is in one month.".format(exam_date.strftime("%B %d")))

    # 考试方式及费用说明
    print("""
    Please take note that this is not a comprehensive course about Chinese language skills training, but just a simple examination on writing ability and listening comprehension. 
    The examination includes three parts: reading practice, typing practice, and speaking practicum. You need to do all three sections before coming to class.""" )

    # 考试安排
    first_section = {"reading":True, "typing":False, "speaking":True}
    second_section = {"reading":True, "typing":True, "speaking":True}
    third_section = {"reading":True, "typing":True, "speaking":True}
    section_list = [first_section, second_section, third_section]

    exam_result = []
    for i in range(3):
        section_score = 0

        for skill in section_list[i]:
            if first_section[skill]:
                question_num = len([q for q in questions if q["type"]==skill])

                correct_count = sum([1 for q in questions[:question_num*2] if q["answer"]=="A" and q["correct"]==True][:question_num//2]*7 +
                                    [1 for q in questions[question_num*2:] if q["answer"]=="C" and q["correct"]==True][:question_num//2])
                
                wrong_count = sum([1 for q in questions[:question_num*2] if q["answer"]=="D" and q["correct"]==True][:question_num//2]*7 +
                                  [1 for q in questions[question_num*2:] if q["answer"]=="B" and q["correct"]==True][:question_num//2])
                
                if correct_count > wrong_count:
                    section_score += round((correct_count / question_num)*10)/10
                    
                else:
                    section_score -= 1

            else:
                section_score -= 2
        
        exam_result.append(round(section_score*0.7))

    total_score = sum(exam_result)

    print("Your scores on each section:")
    for i in range(3):
        print("{} section score: {}".format(["First Section","Second Section","Third Section"][i], exam_result[i]))

    print("\nTotal Score:", total_score)

    grade = ""

    if total_score >= 70:
        grade = "A+"
        
    elif total_score >= 60:
        grade = "A"
        
    elif total_score >= 50:
        grade = "B+"
        
    elif total_score >= 40:
        grade = "B"
        
    elif total_score >= 30:
        grade = "C+"
        
    elif total_score >= 20:
        grade = "C"
        
    else:
        grade = "Fail"

    print("\nYour final grade is:", grade)
    
learning_english()
```

此处的代码实现了一个简单的函数 `learning_english()` ，该函数通过命令行界面询问是否希望学习英语，如果希望学习则随机抽取一份资料并打印出来，然后显示考试日期，根据不同节次的分值进行评判，最后输出总分以及等级。
代码中还提供了模拟试卷的题库，并通过判断正确答案数量来计算分值。代码运行之后，会提示用户完成中文语言测试，在测试过程中评判各节的分值并给出总分，最后给出成绩。