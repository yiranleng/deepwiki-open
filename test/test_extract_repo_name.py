#!/usr/bin/env python3
"""
Focused test script for the _extract_repo_name_from_url method

Run this script to test only the repository name extraction functionality.
Usage: python test_extract_repo_name.py
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add the parent directory to the path to import the data_pipeline module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the modules under test
from api.data_pipeline import DatabaseManager


class TestExtractRepoNameFromUrl:
    """
    _extract_repo_name_from_url 方法的综合测试
    """
    
    def setup_method(self):
        """
        在每个测试方法之前设置测试夹具。
        """
        self.db_manager = DatabaseManager()
    
    def test_extract_repo_name_github_standard_url(self):
        
        # Test standard GitHub URL
        github_url = "https://github.com/owner/repo"
        result = self.db_manager._extract_repo_name_from_url(github_url, "github")
        assert result == "owner_repo"
        
        # Test GitHub URL with .git suffix
        github_url_git = "https://github.com/owner/repo.git"
        result = self.db_manager._extract_repo_name_from_url(github_url_git, "github")
        assert result == "owner_repo"

        # Test GitHub URL with trailing slash
        github_url_slash = "https://github.com/owner/repo/"
        result = self.db_manager._extract_repo_name_from_url(github_url_slash, "github")
        assert result == "owner_repo"
        
        print("✓ GitHub URL tests passed")
    
    def test_extract_repo_name_gitlab_urls(self):
        """
        测试从 GitLab URL 提取仓库名称
        """
        
        # Test standard GitLab URL
        gitlab_url = "https://gitlab.com/owner/repo"
        result = self.db_manager._extract_repo_name_from_url(gitlab_url, "gitlab")
        assert result == "owner_repo"
        
        # Test GitLab URL with subgroups
        gitlab_subgroup = "https://gitlab.com/group/subgroup/repo"
        result = self.db_manager._extract_repo_name_from_url(gitlab_subgroup, "gitlab")
        assert result == "subgroup_repo"
        
        print("✓ GitLab URL tests passed")
    
    def test_extract_repo_name_bitbucket_urls(self):
        """
        测试从 Bitbucket URL 提取仓库名称
        """
        bitbucket_url = "https://bitbucket.org/owner/repo"
        result = self.db_manager._extract_repo_name_from_url(bitbucket_url, "bitbucket")
        assert result == "owner_repo"

        print("✓ Bitbucket URL tests passed")
    
    def test_extract_repo_name_local_paths(self):
        """
        测试从本地路径提取仓库名称
        """
        result = self.db_manager._extract_repo_name_from_url("/home/user/projects/my-repo", "local")
        assert result == "my-repo"

        result = self.db_manager._extract_repo_name_from_url("/var/repos/project.git", "local")
        assert result == "project"

        print("✓ Local path tests passed")

    def test_extract_repo_name_current_implementation_bug(self):
        """
        演示当前实现错误的测试
        """
        # The current implementation references 'type' which is not in scope
        try:
            # This should raise a NameError due to undefined 'type' variable
            result = self.db_manager._extract_repo_name_from_url("https://github.com/owner/repo")
            print("⚠️  WARNING: Expected the current implementation to fail due to undefined 'type' variable")
            print(f"    But got result: {result}")
        except (NameError, TypeError) as e:
            print(f"✓ Current implementation correctly fails with: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error: {type(e).__name__}: {e}")
        
        # Test absolute local path
        local_path = "/home/user/projects/my-repo"
        result = self.db_manager._extract_repo_name_from_url(local_path, "local")
        assert result == "my-repo"
        
        # Test local path with .git suffix
        local_git = "/var/repos/project.git"
        result = self.db_manager._extract_repo_name_from_url(local_git, "local")
        assert result == "project"
        
        print("✓ Local path tests passed")
    
    def test_extract_repo_name_edge_cases(self):
        """
        测试仓库名称提取的边缘情况
        """
        
        # Test URL with insufficient parts (should use fallback)
        short_url = "https://github.com/repo"
        result = self.db_manager._extract_repo_name_from_url(short_url, "github")
        assert result == "repo"
        
        # Test single directory name
        single_name = "my-repo"
        result = self.db_manager._extract_repo_name_from_url(single_name, "local")
        assert result == "my-repo"
        
        print("✓ Edge case tests passed")
