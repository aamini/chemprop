import React, { Component } from 'react';

import { Menu, Icon, Layout } from 'antd';

import TitleLink from './TitleLink';

const Item = Menu.Item;
const { Header } = Layout;

export class NavBar extends Component {
    render() {
        return (
            <Header
                style={{
                    padding: 0,
                    backgroundColor: '#fff',
                    boxShadow: '5px'
                }}
            >
                <TitleLink />
                <Menu
                    mode="horizontal"
                    style={{ marginTop: '10', float: 'right' }}
                >
                    <Item>
                        <a href="#">
                            <Icon type="flag" />
                            Checkpoints
                        </a>
                    </Item>
                    <Item>
                        <a href="#">
                            <Icon type="database" />
                            Datasets
                        </a>
                    </Item>
                    <Item>
                        <a href="#">
                            <Icon type="rise" />
                            Train
                        </a>
                    </Item>
                    <Item>
                        <a href="#">
                            <Icon type="eye" />
                            Predict
                        </a>
                    </Item>
                    <Item>
                        <a href="#">
                            <Icon type="user" />
                            Users
                        </a>
                    </Item>
                </Menu>
            </Header>
        );
    }
}

export default NavBar;
