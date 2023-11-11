                 

# 1.背景介绍


React是一个开源、声明式、高效的前端JavaScript框架，它的设计理念主要聚焦于用户界面（UI）的构建和更新。从15年Facebook推出ReactJS至今已经历经三代版本更新，已成为当下最热门的Web开发框架。

而对于React来说，React元素（element）本身就是一个不可变对象（immutable object）。这是它所擅长处理的数据类型，它通过提供一致性的编程模型与函数式编程范式实现了组件化。

React中有两种类型的组件：受控组件（controlled component）与非受控组件（uncontrolled component）。它们之间的区别可以说是React组件的核心特征之一，也是其实现不同功能的关键。下面就来聊一下这两个组件分别是什么样子的。

- 非受控组件：顾名思义，这类组件的值由外界控制，也就是外部传入的。这种组件只负责展示当前状态，而不对输入进行响应或控制，因此无法实现交互效果。例如在输入框中输入字符并不会影响到输入值。通常情况下，非受控组件会接受用户的输入作为初始化数据，同时将该初始值保存在组件内部，然后渲染出页面。

  ```jsx
  class Input extends Component {
    constructor(props) {
      super(props);
      this.state = {
        inputValue: props.initialValue || ''
      };
    }

    handleChange = event => {
      const value = event.target.value;
      this.setState({ inputValue: value });
    };

    render() {
      return (
        <input
          type="text"
          value={this.state.inputValue}
          onChange={this.handleChange}
        />
      );
    }
  }
  
  // 使用方式
  <Input initialValue={'hello world'}/>
  ```
  
- 受控组件：相对于非受控组件来说，这个名称里包含“受”字，这里指的是组件的值由自己管理，即组件自身状态（state）跟踪用户输入的值。这种组件接收外界传入的值，并根据这个值来渲染页面。当用户改变输入时，组件会更新自己的状态，同时触发重新渲染，使得页面显示出最新的数据。

  ```jsx
  class ControlledInput extends Component {
    state = {
      inputValue: 'hello'
    };
    
    handleInputChange = e => {
      this.setState({
        inputValue: e.target.value
      })
    };
  
    render() {
      return (
        <div>
          <label htmlFor="controlled-input">Controlled Input:</label>
          <input
            id="controlled-input"
            type="text"
            value={this.state.inputValue}
            onChange={this.handleInputChange}
          />
        </div>
      )
    }
  }
  ```

  

  

那么这两种组件之间又有何区别呢？首先，如果组件是通过父组件传值的形式，那么它就是非受控组件；否则，它就是受控组件。接着，非受控组件只有一种状态——用户输入；而受控组件拥有自己的状态，并且可以通过回调函数将状态传递给父组件，从而实现与外界的交互。

另一方面，非受控组件的生命周期简单直观，完全由组件自身决定什么时候渲染，什么时候销毁；而受控组件则需要手工实现一些逻辑，比如去除掉内部的状态，并通过回调函数将用户输入的值保存到父组件状态中。

另外，对于获取输入值，非受控组件只能依赖于DOM事件的回调函数，而受控组件则可以通过React的状态系统获取到用户输入的值。此外，非受控组件中的defaultValue属性可以帮助我们预先设置默认值，但受控组件不能这样做。

综上，非受控组件适用于那些不需要直接管理输入数据的场景，而受控组件则适合需要处理复杂表单逻辑的场景。当然，两者都不是绝对的，总会有各种各样的应用场景。