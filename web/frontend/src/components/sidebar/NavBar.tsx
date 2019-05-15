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
                    boxShadow: '8px 0px 4px #ccc'
                }}
            >
                <TitleLink />
                <Menu
                    mode="horizontal"
                    style={{ float: 'right' }}
                >
                    <Item style={{paddingTop: 8, paddingBottom: 8}}>
                        <a href="#">
                            <Icon type="flag" />
                            Checkpoints
                        </a>
                    </Item>
                    <Item style={{paddingTop: 8, paddingBottom: 8}}>
                        <a href="#">
                            <Icon type="database" />
                            Datasets
                        </a>
                    </Item>
                    <Item style={{paddingTop: 8, paddingBottom: 8}}>
                        <a href="#">
                            <Icon type="rise" />
                            Train
                        </a>
                    </Item>
                    <Item style={{paddingTop: 8, paddingBottom: 8}}>
                        <a href="#">
                            <Icon type="eye" />
                            Predict
                        </a>
                    </Item>
                    <Item style={{paddingTop: 8, paddingBottom: 8}}>
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
