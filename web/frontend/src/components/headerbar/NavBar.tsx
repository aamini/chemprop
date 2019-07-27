import React, { Component } from 'react';

import { Menu, Layout } from 'antd';

import MenuLink from './MenuLink';
import TitleLink from './TitleLink';
import { page, pageName } from '../../api/constants';

const { Header } = Layout;

export interface NavProps {
    changePage: (newPage: pageName) => () => void;
}

export class NavBar extends Component<NavProps> {
    render() {
        return (
            <Header
                style={{
                    paddingLeft: 20,
                    paddingRight: 0,
                    paddingTop: 0,
                    paddingBottom: 0,
                    backgroundColor: '#fff',
                    boxShadow: '8px 0px 4px #ccc',
                    maxHeight: 64
                }}
            >
                <div
                    style={{
                        // maxWidth: page.width,
                        float: 'none',
                        margin: 'auto'
                    }}
                >
                    <TitleLink />
                    <Menu mode="horizontal" style={{ float: 'right' }}>
                        <MenuLink text="Datasets" link="/" icon="database" />
                        <MenuLink text="Checkpoints" link="/" icon="flag" />
                        <MenuLink text="Train" link="/" icon="rise" />
                        <MenuLink text="Predict" link="/" icon="eye" />
                        <MenuLink text="Users" link="/" icon="user" />
                    </Menu>
                </div>
            </Header>
        );
    }
}

export default NavBar;
