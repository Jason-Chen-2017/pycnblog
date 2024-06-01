
作者：禅与计算机程序设计艺术                    

# 1.简介
         

        DeFi（去中心化金融）是一个颠覆性的新领域，它利用区块链技术构建起一个去中心化的去中心化交易所。DeFi的出现将使得各种类型的数字货币流动性得到有效保障，并释放了原本仅仅由中心化机构运营的金融系统的很多潜力。那么如何创建一个DeFi去中心化交易所呢？在本文中，我会用Solidity语言编写一个去中心化交易所。从零到英雄！
        
        在这个项目中，我们将学习以下知识点：
        
        1.什么是Solidity编程语言？
        2.ERC-20代币标准
        3.ERC-721非同质化通证标准
        4.Solidity智能合约编程
        5.Uniswap 演变过程及其代码实现
        6.Dutch auction模型的解释和Solidity实现
        7.WETH 的原理及Solidity实现
        8.如何在Solidity中进行二次代币兑换
        9.如何将Solidity智能合约部署至主网上
        10.扩展阅读
        11.总结
        
        一起来看一下吧！
        
        # 2.Solidity编程语言简介
        ## 2.1 Solidity 是什么？
        Solidity 是一个基于 Ethereum Virtual Machine (EVM) 的高级语言，它被设计用于创建复杂的加密合约。由于Solidity语言具有安全、轻量、高效等特点，因此可以轻松地进行精心的优化。Solidity语言最初由<NAME>在2014年创建，他曾经参与过以太坊的设计和开发工作。在以太坊平台上部署智能合约需要编译成字节码文件，然后通过以太坊客户端发送到网络上执行。由于众多原因导致Solidity社区快速崛起，现在已经成为全球最受欢迎的开发工具之一。
        
        Solidity编译器将Solidity源代码编译为EVM字节码，这些字节码可以运行在Ethereum虚拟机(EVM)上，该虚拟机是由区块链节点执行的计算引擎。EVM是一个Turing Complete的虚拟机，可以执行任何逻辑运算、控制流语句、加密哈希函数等操作。
        
        ### 2.2 为什么要用Solidity？
        首先，Solidity是一种非常安全且轻量级的编程语言，而且具有很强的可移植性。它支持数组、指针、结构体、枚举类型、匿名函数、模块化编程、异常处理机制、元组、类等功能，还可以使用继承和接口机制对代码进行抽象，因此编写智能合约更加方便、高效。Solidity的目标就是为用户提供易于理解、快速部署、低成本地理、面向对象的编程环境。
        
        其次，Solidity具有很好的跨平台特性。因为它是开源语言，所有平台都可以编译生成相同的字节码，所以无论是在Windows、Mac OS X还是Linux系统上，Solidity智能合约都是可以在不同的平台上正常运行的。此外，由于使用了图灵完备的形式，Solidity智能合约具有很强的可读性和可调试性，这对于企业级应用尤其重要。
        
        最后，Solidity相比其他基于EVM的语言具有较高的性能，而且拥有丰富的第三方库支持，例如web3js，使得在区块链上进行各种应用场景的开发变得十分容易。
        
        ### 2.3 Solidity 学习路线图
        下面是Solidity的学习路线图：
        
        1. 熟悉Solidity语法规则。掌握变量定义、数据类型、表达式、条件语句、循环语句、函数、事件、结构体等基本概念。
        2. 使用Solidity官方教程，学习Solidity的基本使用方法。
        3. 了解EVM虚拟机指令集，掌握如何转换Solidity代码到字节码。
        4. 使用外部库，如OpenZeppelin等，实现更多高级特性。
        5. 在实际项目中应用Solidity，搭建自己的去中心化应用。
        
        # 3. ERC-20代币标准
        ## 3.1 ERC-20 简介
        ERC-20 是一个为了使智能合约与非托管代币兼容而制定的标准，允许在区块链上创建具有独特属性的代币，包括：

        1. 发行数量限制；
        2. 分发权限设置；
        3. 代币名称和符号；
        4. 代币余额查询；
        5. 代币转账；
        6. 对代币的允许/冻结；
        7. 代币销毁；

        ERC-20代币标准在2015年10月发布，它描述了如何创建一个符合ERC-20标准的代币。

        ### 3.2 ERC-20 示例
        ```solidity
        pragma solidity ^0.5.0;

        /**
         * @title ERC20 interface
         */
        contract IERC20 {
            function totalSupply() external view returns (uint256);

            function balanceOf(address account) external view returns (uint256);

            function allowance(address owner, address spender) external view returns (uint256);

            function transfer(address recipient, uint256 amount) external returns (bool);

            function approve(address spender, uint256 amount) external returns (bool);

            function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

            event Transfer(address indexed from, address indexed to, uint256 value);

            event Approval(address indexed owner, address indexed spender, uint256 value);
        }

        /**
         * @title Ownable
         * @dev The Ownable contract has an owner address, and provides basic authorization control
         * functions, this simplifies the implementation of "user permissions".
         */
        contract Ownable {
            address private _owner;

            event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

            constructor () internal {
                _owner = msg.sender;
                emit OwnershipTransferred(address(0), _owner);
            }

            function owner() public view returns (address) {
                return _owner;
            }

            modifier onlyOwner() {
                require(msg.sender == _owner, "Ownable: caller is not the owner");
                _;
            }

            function renounceOwnership() public onlyOwner {
                emit OwnershipTransferred(_owner, address(0));
                _owner = address(0);
            }

            function transferOwnership(address newOwner) public onlyOwner {
                _transferOwnership(newOwner);
            }

            function _transferOwnership(address newOwner) internal {
                require(newOwner!= address(0), "Ownable: new owner is the zero address");
                emit OwnershipTransferred(_owner, newOwner);
                _owner = newOwner;
            }
        }

        /**
         * @title SafeMath
         * @dev Math operations with safety checks that throw on error
         */
        library SafeMath {
            function mul(uint256 a, uint256 b) internal pure returns (uint256) {
                if (a == 0) {
                    return 0;
                }

                uint256 c = a * b;
                assert(c / a == b);

                return c;
            }

            function div(uint256 a, uint256 b) internal pure returns (uint256) {
                // assert(b > 0); // Solidity automatically throws when dividing by 0
                uint256 c = a / b;
                // assert(a == b * c + a % b); // There is no case in which this doesn't hold

                return c;
            }

            function sub(uint256 a, uint256 b) internal pure returns (uint256) {
                assert(b <= a);
                return a - b;
            }

            function add(uint256 a, uint256 b) internal pure returns (uint256) {
                uint256 c = a + b;
                assert(c >= a);

                return c;
            }
        }

        /**
         * @title Basic token
         * @dev Basic version of StandardToken, with no allowances.
         */
        contract BasicToken is IERC20, Ownable {
            using SafeMath for uint256;

            mapping(address => uint256) balances;

            uint256 totalSupply_;

            constructor(uint256 total) public {
                totalSupply_ = total;
                balances[msg.sender] = totalSupply_;
            }

            function totalSupply() public override view returns (uint256) {
                return totalSupply_;
            }

            function balanceOf(address account) public override view returns (uint256) {
                return balances[account];
            }

            function transfer(address recipient, uint256 amount) public override returns (bool) {
                _transfer(msg.sender, recipient, amount);
                return true;
            }

            function _transfer(address sender, address recipient, uint256 amount) internal {
                require(sender!= address(0), "ERC20: transfer from the zero address");
                require(recipient!= address(0), "ERC20: transfer to the zero address");

                balances[sender] = balances[sender].sub(amount);
                balances[recipient] = balances[recipient].add(amount);
                emit Transfer(sender, recipient, amount);
            }

            function approve(address spender, uint256 value) public override returns (bool) {
                allowed[msg.sender][spender] = value;
                emit Approval(msg.sender, spender, value);
                return true;
            }

            function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
                _transfer(sender, recipient, amount);
                allowed[sender][msg.sender] = allowed[sender][msg.sender].sub(amount);
                emit Approval(sender, msg.sender, allowed[sender][msg.sender]);
                return true;
            }

            function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
                allowed[msg.sender][spender] = allowed[msg.sender][spender].add(addedValue);
                emit Approval(msg.sender, spender, allowed[msg.sender][spender]);
                return true;
            }

            function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
                uint256 oldValue = allowed[msg.sender][spender];
                if (subtractedValue >= oldValue) {
                    allowed[msg.sender][spender] = 0;
                } else {
                    allowed[msg.sender][spender] = oldValue.sub(subtractedValue);
                }
                emit Approval(msg.sender, spender, allowed[msg.sender][spender]);
                return true;
            }
        }

        /**
         * @title Standard Token
         * @dev Implementation of the EIP20 standard token.
         * https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20.md
         * This contract includes methods for ownership management,
         * approvals, and events.
         * It is based on code by FirstBlood: https://github.com/Firstbloodio/token/blob/master/smart_contract/FirstBloodToken.sol
         * Based on code by Open Zeppelin: https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.3.0/contracts/token/ERC20/ERC20.sol
         * Adapted to follow the ERC20 specification.
         * Rearranged and removed some features.
        */

        contract StandardToken is IERC20, Ownable {
            using SafeMath for uint256;

            mapping(address => uint256) balances;

            mapping(address => mapping (address => uint256)) allowed;

            string public name;
            string public symbol;
            uint8 public decimals;
            uint256 public totalSupply;

            event Transfer(
                address indexed from,
                address indexed to,
                uint256 value
            );

            event Approval(
                address indexed owner,
                address indexed spender,
                uint256 value
            );


            /**
             * @dev Constructor that gives msg.sender all of existing tokens.
             */
            constructor(string memory tokenName,
                        string memory tokenSymbol,
                        uint8 decimalUnits,
                        uint256 initialSupply) public {

                name = tokenName;
                symbol = tokenSymbol;
                decimals = decimalUnits;
                totalSupply = initialSupply * 10 ** uint256(decimals);

                balances[msg.sender] = totalSupply;
                emit Transfer(address(0), msg.sender, totalSupply);
            }


            /**
             * @dev Returns the balance of the specified address.
             * @param owner The address to query the balance of.
             * @return An uint256 representing the amount owned by the passed address.
             */
            function balanceOf(address owner) public override view returns (uint256) {
                return balances[owner];
            }


            /**
             * @dev Transfers tokens from one address to another.
             * Note that while this function emits an Approval event, this is technically
             * not required as per the specification, and other compliant implementations
    		 * may not emit the event.
             * @param from Address to transfer tokens from.
             * @param to Address to transfer tokens to.
             * @param value Amount of tokens to transfer.
             */
            function transfer(address from,
                              address to,
                              uint256 value) public override returns (bool) {

                _transfer(from, to, value);
                return true;
            }


            /**
             * @dev Internal function that mints an amount of the token and assigns it to
             * an account. This encapsulates the modification of balances such that the
             * proper events are emitted.
             * @param account The account that will receive the created tokens.
             * @param value The amount that will be created.
             */
            function _mint(address account, uint256 value) internal {
                require(account!= address(0), "ERC20: mint to the zero address");

                totalSupply = totalSupply.add(value);
                balances[account] = balances[account].add(value);
                emit Transfer(address(0), account, value);
            }


            /**
             * @dev Internal function that burns an amount of the token of a given
    		 * account.
             * @param account The account whose tokens will be burnt.
             * @param value The amount that will be burnt.
             */
            function _burn(address account, uint256 value) internal {
                require(account!= address(0), "ERC20: burn from the zero address");

                balances[account] = balances[account].sub(value);
                totalSupply = totalSupply.sub(value);
                emit Transfer(account, address(0), value);
            }


            /**
             * @dev Internal function that transfers tokens between two accounts.
             * Emits appropriate events.
             * @param from Account to transfer from.
             * @param to Account to transfer to.
             * @param value Amount to transfer.
             */
            function _transfer(address from,
                               address to,
                               uint256 value) internal {

                require(to!= address(0), "ERC20: transfer to the zero address");

                balances[from] = balances[from].sub(value);
                balances[to] = balances[to].add(value);
                emit Transfer(from, to, value);
            }


            /**
             * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
             * Beware that changing an allowance with this method brings the risk that someone may use both the old
             * and the new allowance by unfortunate transaction ordering. One possible solution to mitigate this
             * race condition is to first reduce the spender's allowance to 0 and set the desired value afterwards:
             * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
             * @param spender The address which will spend the funds.
             * @param value The amount of tokens to be spent.
             */
            function approve(address spender, uint256 value) public override returns (bool) {
                allowed[msg.sender][spender] = value;
                emit Approval(msg.sender, spender, value);
                return true;
            }


            /**
             * @dev Function to check the amount of tokens that an owner allowed to a spender.
             * @param owner address The address which owns the funds.
             * @param spender address The address which will spend the funds.
             * @return A uint256 specifying the amount of tokens still available for the spender.
             */
            function allowance(address owner, address spender) public override view returns (uint256) {
                return allowed[owner][spender];
            }


            /**
             * @dev Increase the amount of tokens that an owner allowed to a spender.
             * approve should be called when allowed_[_spender] == 0. To increment
             * allowed value is better to use this function to avoid 2 calls (and wait until
             * the first transaction is mined)
             * From MonolithDAO Token.sol
             * Emits an Approval event.
             * @param spender The address which will spend the funds.
             * @param addedValue The amount of tokens to increase the allowance by.
             */
            function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
                allowed[msg.sender][spender] = allowed[msg.sender][spender].add(addedValue);
                emit Approval(msg.sender, spender, allowed[msg.sender][spender]);
                return true;
            }


            /**
             * @dev Decrease the amount of tokens that an owner allowed to a spender.
             * approve should be called when allowed_[_spender] == 0. To decrement
             * allowed value is better to use this function to avoid 2 calls (and wait until
             * the first transaction is mined)
             * From MonolithDAO Token.sol
             * Emits an Approval event.
             * @param spender The address which will spend the funds.
             * @param subtractedValue The amount of tokens to decrease the allowance by.
             */
            function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
                uint256 currentAllowance = allowed[msg.sender][spender];
                if (currentAllowance < subtractedValue) {
                    allowed[msg.sender][spender] = 0;
                } else {
                    allowed[msg.sender][spender] = currentAllowance.sub(subtractedValue);
                }
                emit Approval(msg.sender, spender, allowed[msg.sender][spender]);
                return true;
            }


        }
        
        ```

      